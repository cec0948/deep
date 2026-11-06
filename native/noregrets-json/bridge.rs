// Public replay and trusted local evaluation over the pinned rules engine.
use engine::{Hand, HandConfig, PlayerAction};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};
use serde_json::{json, Value};
use std::{io::{self, BufRead, Write}, sync::Arc};
fn action(h:&Hand,a:&Value)->Result<PlayerAction,String>{
    if a["seat"].as_u64()!=Some(h.to_act() as u64){return Err("wrong acting seat".into())}
    match a["kind"].as_str(){
        Some("check") if h.can_check()=>Ok(PlayerAction::CheckCall),
        Some("call") if !h.can_check()=>Ok(PlayerAction::CheckCall),
        Some("fold") if !h.can_check()=>Ok(PlayerAction::Fold),
        Some("raise")=>{let n=a["to"].as_u64().filter(|&x|x<=u32::MAX as u64).ok_or("invalid raise")? as u32;
            let (lo,hi)=h.raise_bounds().ok_or("raise unavailable")?;if n<lo||n>hi{return Err("raise outside bounds".into())}Ok(PlayerAction::RaiseTo(n))},
        _=>Err("illegal action".into())
    }
}
fn observe(h:&Hand, log:&Vec<Value>)->Value{
    if h.is_terminal(){return json!({"terminal":true,"utilities":h.utilities()})}
    let p=h.to_act();let bounds=h.raise_bounds();
    json!({"terminal":false,"your_cards":h.hole(p).map(cards::card_str),"community_cards":h.board().iter().map(|&c|cards::card_str(c)).collect::<Vec<_>>(),
        "dealer":h.button(),"seat_to_act":p,"street":format!("{:?}",h.street()).to_lowercase(),"pot":h.pot(),"your_stack":h.stack(p),
        "your_bet_this_street":h.street_commit(p),"amount_owed":h.to_call(),"current_bet":h.current_bet(),"min_raise_to":bounds.map(|b|b.0),
        "raise_bounds":bounds,"can_check":h.can_check(),"players":(0..6).map(|i|json!({"seat":i,"stack":h.stack(i),"bet_this_street":h.street_commit(i),"is_folded":h.folded(i),"is_all_in":h.all_in(i)})).collect::<Vec<_>>(),"action_log":log})
}
fn log_start(button:usize)->Vec<Value>{vec![json!({"seat":(button+1)%6,"action":"small_blind","amount":50}),json!({"seat":(button+2)%6,"action":"big_blind","amount":100})]}
fn apply(h:&mut Hand,log:&mut Vec<Value>,a:&Value)->Result<PlayerAction,String>{
    if h.is_terminal(){return Err("terminal hand".into())}let act=action(h,a)?;let old=h.street();
    // Fullhouse call logs contain incremental chips; raises contain street totals.
    log.push(json!({"seat":h.to_act(),"action":a["kind"],"amount":a.get("to").cloned().unwrap_or(json!(if a["kind"]=="call" {h.to_call().min(h.stack(h.to_act()))}else{0}))}));
    h.apply(act);if !h.is_terminal()&&h.street()!=old{log.push(json!({"action":"deal"}))}Ok(act)
}
fn main()->Result<(),Box<dyn std::error::Error>>{
    let args:Vec<_>=std::env::args().collect();
    let policy=if let Some(path)=args.get(1){let bp=cfr::Blueprint::load(path)?;if bp.num_players!=6{return Err("six-player model required".into())}
        let abs=abstraction::Abstraction::with_centroids(bp.abs_cfg.clone(),bp.centroids.clone());Some(bot::Policy::new(bp,Arc::new(abs)))}else{None};
    let mut game:Option<table::Table>=None;let mut log=vec![];
    let mut output=io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines(){let line=line?;
        let result=(||->Result<Value,String>{
            let r:Value=serde_json::from_str(&line).map_err(|e|e.to_string())?;
            match r["op"].as_str().unwrap_or("replay"){
                "replay"=>{
                    let hero=r["hero"].as_u64().filter(|&x|x<6).ok_or("invalid hero")? as usize;
                    let button=r["button"].as_u64().filter(|&x|x<6).ok_or("invalid button")? as usize;
                    let hole=r["hole"].as_array().filter(|x|x.len()==2).ok_or("hole needs two cards")?;
                    let board=r["board"].as_array().cloned().unwrap_or_default();if ![0,3,4,5].contains(&board.len()){return Err("invalid board length".into())}
                    let mut known=vec![];for s in hole.iter().chain(board.iter()){let c=cards::parse_card(s.as_str().ok_or("invalid card")?).ok_or("invalid card")?;if known.contains(&c){return Err("duplicate card".into())}known.push(c)}
                    let mut deck=[255;52];deck[hero*2..hero*2+2].copy_from_slice(&known[..2]);deck[12..12+board.len()].copy_from_slice(&known[2..]);
                    let mut rest:Vec<u8>=(0..52).filter(|c|!known.contains(c)).collect();if r["placeholder_seed"].is_u64(){rest.shuffle(&mut SmallRng::seed_from_u64(r["placeholder_seed"].as_u64().unwrap()))}
                    let mut it=rest.into_iter();for c in &mut deck{if *c==255{*c=it.next().unwrap()}}
                    let mut h=Hand::new(&HandConfig::default(),button,deck);let mut l=log_start(button);
                    if let Some(actions)=r["actions"].as_array(){if actions.len()>512{return Err("too many actions".into())}for a in actions{apply(&mut h,&mut l,a)?;}}
                    if h.is_terminal()||h.to_act()!=hero||h.board()!=&known[2..]{return Err("hero/board inconsistent with replay".into())}Ok(observe(&h,&l))
                },
                "reset"=>{let mut deck=cards::fresh_deck();deck.shuffle(&mut SmallRng::seed_from_u64(r["seed"].as_u64().ok_or("seed required")?));let b=r["button"].as_u64().filter(|&x|x<6).ok_or("button required")? as usize;
                    game=Some(table::Table::new(&HandConfig::default(),b,deck));log=log_start(b);Ok(observe(&game.as_ref().unwrap().real,&log))},
                "step"=>{let t=game.as_mut().ok_or("reset required")?;let mut copy=t.real.clone();let a=apply(&mut copy,&mut log,&r["action"])?;
                    if let Some(p)=&policy{t.apply_concrete(a,&p.abs)}else{t.real=copy}Ok(observe(&t.real,&log))},
                "blueprint"=>{let t=game.as_ref().ok_or("reset required")?;let p=policy.as_ref().ok_or("model required")?;let h=&t.real;if h.is_terminal(){return Err("terminal".into())}
                    let mut rng=SmallRng::seed_from_u64(r["seed"].as_u64().unwrap_or(0));let before=bot::fallback_stats();let (aa,pr)=p.blueprint_dist(&t.shadow,&t.hist,&mut rng);let i=cfr::sample_index(&pr,&mut rng);let a=p.abs.concrete(h,aa[i]);
                    let a=match a{PlayerAction::Fold=>json!({"kind":"fold"}),PlayerAction::CheckCall=>json!({"kind":if h.can_check(){"check"}else{"call"}}),PlayerAction::RaiseTo(n)=>json!({"kind":"raise","to":n})};
                    Ok(json!({"action":a,"fallback":bot::fallback_stats().iter().zip(before).any(|(a,b)|a.1>b.1)}))},
                _=>Err("unknown operation".into())
            }
        })();
        let v=result.map(|v|json!({"ok":true,"state":v})).unwrap_or_else(|e|json!({"ok":false,"error":e}));serde_json::to_writer(&mut output,&v)?;writeln!(output)?;output.flush()?;
    }Ok(())
}
