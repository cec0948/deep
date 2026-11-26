// Included beside the frozen terminal replay, common-history and action-transition readers.
pub fn pool528() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let start: u64 = args[4].parse()?;
    let deals: usize = args[5].parse()?;
    let cfg = HandConfig::default();
    let mut tapes = std::io::stdin().lock();
    let mut feeds = args[1..4].iter().map(|p| read_pipe(p)).collect::<Result<Vec<_>,_>>()?;
    let pairs = [(2,0), (1,0), (2,1)];
    let mut counts = [[0u64; 6]; 3];
    let mut transitions: [[BTreeMap<String, (u64,i64)>;4];3] = Default::default();
    let mut outputs = Vec::with_capacity(deals);
    for deal in 0..deals {
        let mut payoffs = [0i64;3];
        let mut gains = [[0i64;6];3];
        for seat in 0..6 {
            let mut tape = [0u8;9];
            std::io::Read::read_exact(&mut tapes, &mut tape)?;
            let expected_seed = u64::from_be_bytes(tape[0..8].try_into()?);
            let slot = usize::from(tape[8]);
            if slot >= 4 { return Err("unregistered slot".into()); }
            let iterations = [None, Some(486 + 6 * slot), Some(510 + 6 * slot)];
            let mut hands = Vec::with_capacity(3);
            for (p,(_,reader)) in feeds.iter_mut().enumerate() {
                let mut line = String::new();
                reader.read_line(&mut line)?;
                let row: ComparedHand = serde_json::from_str(&line)?;
                if row.deal != deal || row.hand.hero != seat || row.hand.seed != start + deal as u64 ||
                    row.hand_seed != expected_seed || row.snapshot_iteration != iterations[p] {
                    return Err("wrong deal, seat, hand RNG or private snapshot choice".into());
                }
                validate(&row.hand, &cfg)?;
                payoffs[p] += row.hand.chips;
                hands.push(row.hand);
            }
            for (j, &(a,b)) in pairs.iter().enumerate() {
                let (candidate, reference) = (&hands[a], &hands[b]);
                let (category,index) = divergence(candidate,reference,&cfg)?;
                counts[j][category] += 1;
                let delta = candidate.chips - reference.chips;
                gains[j][category] += delta;
                if category < 4 {
                    let i = index.ok_or("missing changed decision")?;
                    let key = transition(&reference.actions[i], &candidate.actions[i])?;
                    let entry = transitions[j][category].entry(key).or_default();
                    entry.0 += 1;
                    entry.1 += delta;
                }
            }
        }
        outputs.push((payoffs,gains));
    }
    if !tapes.fill_buf()?.is_empty() {return Err("extra private tape".into());}
    for (child,mut reader) in feeds {
        if reader.read_line(&mut String::new())? != 0 || !child.wait_with_output()?.status.success() {
            return Err("extra or corrupt hands".into());
        }
    }
    println!("{}",serde_json::json!({"deals": outputs, "counts": counts, "transitions": transitions}));
    Ok(())
}
