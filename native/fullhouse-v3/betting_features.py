def _replay_betting_rounds(log, players, current_street_name):
    """Full-raise counts and last actual bettor; explicit street markers preferred.

    Calls are incremental; raises are street totals. Inferring streets without
    markers is supported only for the fixed six-player 100BB contract.
    """
    target=_street_index(current_street_name)
    seats={int(p.get('seat',i)) for i,p in enumerate(players)}
    street=0;cur=0;minimum=BIG_BLIND;bets={};invested={};folded=set();allin=set()
    pending=set(seats);segment=[];raises=0;raiser=None

    def active():
        return seats-folded-allin

    def advance():
        nonlocal street,cur,minimum,bets,pending,segment,raises,raiser
        street+=1;cur=0;minimum=BIG_BLIND;bets={};pending=active()
        segment=[];raises=0;raiser=None

    for entry in log:
        kind=entry.get('action')
        if kind=='deal':
            advance();continue
        if entry.get('seat') is None:continue
        seat=int(entry['seat']);amount=max(0,_safe_int(entry.get('amount',0)))
        if kind not in ('small_blind','big_blind') and not pending and street<target:
            advance()
        if street==target:segment.append(entry)
        inc=0
        if kind in ('small_blind','big_blind','call'):
            inc=amount;bets[seat]=bets.get(seat,0)+inc
            if kind!='call':cur=max(cur,bets[seat])
            else:pending.discard(seat)
        elif kind in ('raise','all_in'):
            inc=max(0,amount-bets.get(seat,0));bets[seat]=amount
            raise_size=max(0,amount-cur);cur=max(cur,amount);pending.discard(seat)
            if raise_size>0:
                raiser=seat
                if raise_size>=minimum:
                    minimum=raise_size;raises+=1;pending=active()-{seat}
                # A short raise must still be answered by earlier callers.
                pending |= {p for p in active() if p!=seat and bets.get(p,0)<cur}
        elif kind=='fold':
            folded.add(seat);pending.discard(seat)
        elif kind=='check':pending.discard(seat)
        if inc:
            invested[seat]=invested.get(seat,0)+inc
            if invested[seat]>=STARTING_STACK:allin.add(seat);pending.discard(seat)
        if kind=='all_in':allin.add(seat);pending.discard(seat)
    if street<target and not pending:advance()
    return segment,invested,min(raises,4),raiser,minimum


def _current_street_aggression(log, players, current_bet, street_name):
    _,_,raises,raiser,minimum=_replay_betting_rounds(log,players,street_name)
    return raises,raiser,minimum


def _hand_invested_from_log(log,players=None,street_name='preflop'):
    return _replay_betting_rounds(log,players or [],street_name)[1]
