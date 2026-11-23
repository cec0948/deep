// Included in the same module as the frozen terminal log reader.
use std::io::BufReader;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct ComparedHand {
    #[serde(flatten)]
    hand: LoggedHand,
    hand_seed: u64,
    deal: usize,
    snapshot_iteration: Option<usize>,
}

fn read_pipe(path: &str) -> std::io::Result<(Child, BufReader<ChildStdout>)> {
    let mut child = Command::new("gzip").args(["-dc", path]).stdout(Stdio::piped()).spawn()?;
    let reader = BufReader::new(child.stdout.take().unwrap());
    Ok((child,reader))
}

fn divergence(a: &LoggedHand, b: &LoggedHand, cfg: &HandConfig) -> Result<(usize, Option<usize>), &'static str> {
    let same = |x: &LoggedAction,y: &LoggedAction| x.kind==y.kind && x.to==y.to && x.seat==y.seat;
    let first = a.actions.iter().zip(&b.actions).position(|(x,y)| !same(x,y));
    let Some(idx) = first else {
        if a.actions.len()!=b.actions.len() || a.utilities!=b.utilities { return Err("unmatched strict prefix or payoff"); }
        return Ok((5, None));
    };
    let mut deck = cards::fresh_deck();
    deck.shuffle(&mut SmallRng::seed_from_u64(a.seed));
    let mut hand = Hand::new(cfg,0,deck);
    for action in a.actions.iter().take(idx) {
        let mv = match (action.kind.as_str(),action.to) {
            ("fold",None) => PlayerAction::Fold,
            ("check"|"call",None) => PlayerAction::CheckCall,
            ("raise",Some(to)) => PlayerAction::RaiseTo(to),
            _ => return Err("bad common action"),
        };
        if hand.is_terminal() || hand.to_act()!=action.seat { return Err("invalid shared prefix"); }
        hand.apply(mv);
    }
    if hand.is_terminal() || hand.to_act()!=a.actions[idx].seat || hand.to_act()!=b.actions[idx].seat {
        return Err("action taken from inconsistent public history");
    }
    Ok((if hand.to_act()==a.hero {hand.street() as usize} else {4}, Some(idx)))
}

fn transition(reference: &LoggedAction, candidate: &LoggedAction) -> Result<String, &'static str> {
    if reference.kind == "raise" && candidate.kind == "raise" {
        let old = reference.to.ok_or("missing reference raise amount")?;
        let new = candidate.to.ok_or("missing candidate raise amount")?;
        if new == old { return Err("identical raise at alleged first divergence"); }
        return Ok(if new > old { "raise_to_higher_raise" } else { "raise_to_lower_raise" }.to_owned());
    }
    if reference.kind == candidate.kind { return Err("same nonraise at alleged first divergence"); }
    Ok(format!("{}_to_{}", reference.kind, candidate.kind))
}

pub fn aggregate() -> Result<(),Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let paths = &args[1..4];
    let start: u64 = args[4].parse()?;
    let deals: usize = args[5].parse()?;
    let mut feeds = paths.iter().map(|p| read_pipe(p)).collect::<Result<Vec<_>,_>>()?;
    let mut counts = [[0u64;6];3];
    let mut chips = [[0i64;6];3];
    let mut actual = [0i64;3];
    let mut flop_transitions = [BTreeMap::<String,(u64,i64)>::new(), BTreeMap::new(), BTreeMap::new()];
    let cfg=HandConfig::default();
    let pairs=[(2,0),(1,0),(2,1)];
    for i in 0..deals*6 {
        let mut hands = Vec::with_capacity(3);
        for (p,(_,reader)) in feeds.iter_mut().enumerate() {
            let mut line = String::new();
            reader.read_line(&mut line)?;
            let h: ComparedHand = serde_json::from_str(&line)?;
            if h.deal!=i/6 || h.hand.hero!=i%6 || h.hand.seed!=start+h.deal as u64 {return Err("wrong card/seat order".into());}
            validate(&h.hand,&cfg)?;
            actual[p]+=h.hand.chips;
            hands.push(h);
        }
        if hands[0].hand_seed!=hands[1].hand_seed || hands[0].hand_seed!=hands[2].hand_seed ||
            hands[0].snapshot_iteration.is_some() || hands[1].snapshot_iteration.is_none() ||
            hands[1].snapshot_iteration!=hands[2].snapshot_iteration {return Err("unmatched paired tapes".into());}
        for (j,(a,b)) in pairs.iter().enumerate() {
            let l=&hands[*a].hand;
            let r=&hands[*b].hand;
            let (category, index)=divergence(l,r,&cfg)?;
            counts[j][category]+=1;
            chips[j][category]+=l.chips-r.chips;
            if category == 1 {
                let index = index.ok_or("unmatched flop action")?;
                let key = transition(&r.actions[index], &l.actions[index])?;
                let totals = flop_transitions[j].entry(key).or_default();
                totals.0 += 1;
                totals.1 += l.chips-r.chips;
            }
        }
    }
    for (child,mut reader) in feeds {
        if reader.read_line(&mut String::new())?!=0 || !child.wait_with_output()?.status.success() {
            return Err("extra or corrupt hands".into());
        }
    }
    println!("{}",serde_json::json!({"counts":counts,"chip_differences":chips,"policy_chips":actual,
                                     "flop_transitions":flop_transitions}));
    Ok(())
}
