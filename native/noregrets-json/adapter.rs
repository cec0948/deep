// Thin transport over the pinned upstream engine. No training/game algorithm here.
use rand::{rngs::SmallRng, SeedableRng};
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use engine::{HandConfig, PlayerAction};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    hero: usize,
    button: usize,
    hole: [String; 2],
    #[serde(default)]
    board: Vec<String>,
    #[serde(default)]
    actions: Vec<Action>,
    #[serde(default)]
    seed: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Action {
    seat: usize,
    kind: String,
    to: Option<u32>,
}

fn action_json(a: PlayerAction, h: &engine::Hand) -> Value {
    match a {
        PlayerAction::Fold => json!({"kind":"fold"}),
        PlayerAction::CheckCall => json!({"kind":if h.can_check() {"check"} else {"call"}}),
        PlayerAction::RaiseTo(n) => json!({"kind":"raise","to":n}),
    }
}

fn decide(policy: &bot::Policy, r: Request) -> Result<Value, String> {
    if r.hero >= 6 || r.button >= 6 || ![0,3,4,5].contains(&r.board.len()) {
        return Err("hero/button must be 0..5 and board must have 0,3,4,5 cards".into());
    }
    if r.actions.len() > 512 { return Err("too many actions".into()); }
    // Only hero's private cards and the public board enter this boundary.
    let known: Vec<u8> = r.hole.iter().chain(&r.board)
        .map(|s| cards::parse_card(s).ok_or_else(|| format!("invalid card: {s}")))
        .collect::<Result<_,_>>()?;
    let mut used = [false;52];
    for &c in &known {
        if used[c as usize] { return Err("duplicate card".into()); }
        used[c as usize] = true;
    }
    // Unknown cards are inert placeholders. Blueprint lookup reads only the acting
    // player's hole cards and public board; online search is deliberately absent.
    let mut deck = [255u8;52];
    deck[2*r.hero] = known[0];
    deck[2*r.hero+1] = known[1];
    deck[12..12+r.board.len()].copy_from_slice(&known[2..]);
    let mut rest = (0..52u8).filter(|&c| !used[c as usize]);
    for c in &mut deck { if *c == 255 { *c = rest.next().unwrap(); } }
    let mut table = table::Table::new(&HandConfig::default(), r.button, deck);
    for a in &r.actions {
        let h = &table.real;
        if h.is_terminal() { return Err("action after terminal hand".into()); }
        if h.to_act() != a.seat { return Err(format!("expected seat {}, got {}", h.to_act(), a.seat)); }
        let concrete = match (a.kind.as_str(), a.to) {
            ("check", None) if h.can_check() => PlayerAction::CheckCall,
            ("call", None) if !h.can_check() => PlayerAction::CheckCall,
            ("fold", None) if !h.can_check() => PlayerAction::Fold,
            ("raise", Some(to)) => {
                let (lo,hi) = h.raise_bounds().ok_or("raising is unavailable")?;
                if !(lo..=hi).contains(&to) { return Err(format!("raise-to must be in {lo}..={hi}")); }
                PlayerAction::RaiseTo(to)
            }
            _ => return Err("invalid action or inconsistent to amount".into()),
        };
        table.apply_concrete(concrete, &policy.abs);
    }
    let h = &table.real;
    if h.is_terminal() { return Err("hand is terminal".into()); }
    if h.to_act() != r.hero { return Err("it is not hero's turn".into()); }
    if h.board() != &known[2..] { return Err("board does not match the replayed betting street".into()); }
    let before = bot::fallback_stats();
    let mismatches = bot::menu_mismatch_count();
    let mut rng = SmallRng::seed_from_u64(r.seed);
    let (acts, probs) = policy.blueprint_dist(&table.shadow, &table.hist, &mut rng);
    if probs.iter().any(|p| !p.is_finite() || *p < 0.) || (probs.iter().sum::<f64>()-1.).abs() > 1e-4 {
        return Err("model returned invalid probabilities".into());
    }
    let concrete: Vec<_> = acts.iter().map(|&a| {
        if h.raise_bounds().is_none() && matches!(a, abstraction::AbsAction::Bet(_) | abstraction::AbsAction::AllIn) {
            PlayerAction::CheckCall
        } else { policy.abs.concrete(h,a) }
    }).collect();
    let after = bot::fallback_stats();
    let fallback = after.iter().zip(before).any(|(a,b)| a.1 > b.1);
    let source = if bot::menu_mismatch_count() > mismatches {"menu_mismatch_fallback"}
        else if fallback {"unseen_or_zero_mass_fallback"} else {"blueprint"};
    let chosen = cfr::sample_index(&probs, &mut rng);
    let distribution: Vec<_> = concrete.iter().zip(&probs).map(|(&a,p)| {
        json!({"action":action_json(a,h),"probability":p})
    }).collect();
    Ok(json!({"ok":true,"action":action_json(concrete[chosen],h),
        "distribution":distribution,"policy_source":source,"strength_verified":false,
        "model_iterations":policy.blueprint.iterations,"players":6,"starting_stack_bb":100,
        "pot":h.pot(),"to_call":h.to_call(),"raise_bounds":h.raise_bounds(),
        "street":format!("{:?}",h.street())}))
}

fn run() -> Result<(),Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 { return Err("usage: poker-json MODEL.bin".into()); }
    let bp = cfr::Blueprint::load(&args[1])?;
    if bp.num_players != 6 { return Err("a six-player blueprint is required".into()); }
    if bp.centroids.as_ref().is_some_and(|c| c.is_strategic()) {
        return Err("strategic-clustering models require predecessor context and are unsupported".into());
    }
    let abs = abstraction::Abstraction::with_centroids(bp.abs_cfg.clone(),bp.centroids.clone());
    let policy = bot::Policy::new(bp, Arc::new(abs));
    let mut output = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let line = line?;
        let result = if line.len() > 1_048_576 {Err("request exceeds 1 MiB".to_string())}
            else {serde_json::from_str(&line).map_err(|e| e.to_string()).and_then(|r| decide(&policy,r))};
        let response = result.unwrap_or_else(|e| json!({"ok":false,"error":e}));
        serde_json::to_writer(&mut output,&response)?;
        writeln!(output)?;
        output.flush()?;
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() { eprintln!("{e}"); std::process::exit(1); }
}
