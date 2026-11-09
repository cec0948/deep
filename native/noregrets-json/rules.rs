// Fixed-100BB differential test oracle over the pinned, independent Rust engine.
use engine::{Hand, HandConfig, PlayerAction};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

fn observe(h: &Hand, log: &[Value]) -> Value {
    if h.is_terminal() { return json!({"terminal":true,"utilities":h.utilities()}); }
    let p=h.to_act();let bounds=h.raise_bounds();
    json!({"terminal":false,"your_cards":h.hole(p).map(cards::card_str),
        "community_cards":h.board().iter().map(|&c|cards::card_str(c)).collect::<Vec<_>>(),
        "dealer":h.button(),"seat_to_act":p,"street":format!("{:?}",h.street()).to_lowercase(),
        "pot":h.pot(),"your_stack":h.stack(p),"your_bet_this_street":h.street_commit(p),
        "amount_owed":h.to_call(),"current_bet":h.current_bet(),"min_raise_to":bounds.map(|b|b.0),
        "raise_bounds":bounds,"can_check":h.can_check(),
        "players":(0..6).map(|i|json!({"seat":i,"stack":h.stack(i),"bet_this_street":h.street_commit(i),
            "hand_commit":h.hand_commit(i),"is_folded":h.folded(i),"is_all_in":h.all_in(i)})).collect::<Vec<_>>(),"action_log":log})
}

fn main() -> Result<(),Box<dyn std::error::Error>> {
    let mut hand:Option<Hand>=None;let mut log=Vec::new();
    let mut output=io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let result=(|| -> Result<Value,String> {
            let r:Value=serde_json::from_str(&line.as_ref().map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            match r["op"].as_str() {
                Some("reset") => {
                    let b=r["button"].as_u64().filter(|&x|x<6).ok_or("button")? as usize;
                    let values=r["deck"].as_array().filter(|x|x.len()==52).ok_or("52 cards required")?;
                    let mut deck=[0u8;52];let mut seen=[false;52];
                    for (i,s) in values.iter().enumerate() {
                        let c=cards::parse_card(s.as_str().ok_or("card")?).ok_or("card")?;
                        if seen[c as usize] {return Err("duplicate card".into())}
                        seen[c as usize]=true;deck[i]=c;
                    }
                    hand=Some(Hand::new(&HandConfig::default(),b,deck));
                    log=vec![json!({"seat":(b+1)%6,"action":"small_blind","amount":50}),json!({"seat":(b+2)%6,"action":"big_blind","amount":100})];
                },
                Some("step") => {
                    let h=hand.as_mut().ok_or("reset required")?;
                    if h.is_terminal() {return Err("terminal".into())}
                    let a=&r["action"];
                    if a["seat"].as_u64()!=Some(h.to_act() as u64) {return Err("wrong actor".into())}
                    let action=match a["kind"].as_str() {
                        Some("fold") if !h.can_check()=>PlayerAction::Fold,
                        Some("check") if h.can_check()=>PlayerAction::CheckCall,
                        Some("call") if !h.can_check()=>PlayerAction::CheckCall,
                        Some("raise")=>{
                            let to=a["to"].as_u64().filter(|&x|x<=u32::MAX as u64).ok_or("raise")? as u32;
                            let (lo,hi)=h.raise_bounds().ok_or("raise unavailable")?;
                            if to<lo||to>hi {return Err("raise bounds".into())}
                            PlayerAction::RaiseTo(to)
                        },
                        _=>return Err("illegal action".into())
                    };
                    let amount=match action {PlayerAction::RaiseTo(n)=>n,PlayerAction::CheckCall=>h.to_call().min(h.stack(h.to_act())),_=>0};
                    log.push(json!({"seat":h.to_act(),"action":a["kind"],"amount":amount}));
                    let before=h.street();h.apply(action);
                    if !h.is_terminal()&&h.street()!=before {log.push(json!({"action":"deal"}))}
                },
                _=>return Err("unknown operation".into())
            }
            Ok(observe(hand.as_ref().ok_or("reset required")?,&log))
        })();
        let value=result.map(|state|json!({"ok":true,"state":state})).unwrap_or_else(|e|json!({"ok":false,"error":e}));
        serde_json::to_writer(&mut output,&value)?;writeln!(output)?;output.flush()?;
    }
    Ok(())
}
