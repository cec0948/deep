// Emits one checksum-independent count for a stream of evaluated, seeded hands.
use engine::{Hand, HandConfig, PlayerAction};
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
use serde::Deserialize;
use std::io::{self, BufRead};

#[derive(Deserialize)]
struct LoggedAction {
    seat: usize,
    kind: String,
    to: Option<u32>,
}

#[derive(Deserialize)]
struct LoggedHand {
    seed: u64,
    hero: usize,
    chips: i64,
    steps: usize,
    actions: Vec<LoggedAction>,
    utilities: [i64; 6],
}

fn validate(row: &LoggedHand, config: &HandConfig) -> Result<(), &'static str> {
    if row.steps != row.actions.len() || row.steps > 512 || row.hero >= 6 {
        return Err("invalid hero or action count");
    }
    let mut deck = cards::fresh_deck();
    deck.shuffle(&mut SmallRng::seed_from_u64(row.seed));
    let mut hand = Hand::new(config, 0, deck);
    for action in &row.actions {
        if hand.is_terminal() || hand.to_act() != action.seat {
            return Err("wrong action seat or terminal prefix");
        }
        let legal = match (action.kind.as_str(), action.to) {
            ("check", None) if hand.can_check() => PlayerAction::CheckCall,
            ("call", None) if !hand.can_check() => PlayerAction::CheckCall,
            ("fold", None) if !hand.can_check() => PlayerAction::Fold,
            ("raise", Some(chips)) => {
                let (lo, hi) = hand.raise_bounds().ok_or("raise unavailable")?;
                if !(lo..=hi).contains(&chips) {
                    return Err("raise outside bounds");
                }
                PlayerAction::RaiseTo(chips)
            }
            _ => return Err("illegal action"),
        };
        hand.apply(legal);
    }
    if !hand.is_terminal() || hand.utilities() != row.utilities || row.chips != row.utilities[row.hero] {
        return Err("terminal payout mismatch");
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = HandConfig::default();
    let mut hands = 0u64;
    let mut decisions = 0u64;
    for line in io::stdin().lock().lines() {
        let row: LoggedHand = serde_json::from_str(&line?)?;
        validate(&row, &config).map_err(|reason| format!("hand {hands}, card {}, hero {}: {reason}", row.seed, row.hero))?;
        hands += 1;
        decisions += row.actions.len() as u64;
    }
    println!("{}", serde_json::json!({"hands": hands, "decisions": decisions}));
    Ok(())
}
