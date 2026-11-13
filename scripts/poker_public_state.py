"""Lossless public poker observations and explicitly approximate marginal ranges.

The betting ledger mirrors the pinned six-seat,100BB Bridge observation contract.
It is an observation validator/adapter, not an alternative showdown engine. No
opponent private cards or undealt board cards are accepted or synthesized.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations, permutations
import json
import math
from typing import Callable, Iterable, Mapping

RANKS = '23456789TJQKA'
SUITS = 'cdhs'
CARDS = tuple(r+s for s in SUITS for r in RANKS)
STREETS = ('preflop', 'flop', 'turn', 'river')
BOARD_LENGTHS = (0, 3, 4, 5)
STARTING_STACK = 10000
BIG_BLIND = 100


def _integer(value, label, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError('Invalid '+label)
    return value


def _cards(values, count=None):
    if not isinstance(values, (list, tuple)) or (count is not None and len(values) != count):
        raise ValueError('Invalid card collection')
    result = tuple(values)
    if any(not isinstance(c, str) or len(c) != 2 or c[0] not in RANKS or c[1] not in SUITS for c in result):
        raise ValueError('Invalid card')
    if len(set(result)) != len(result):
        raise ValueError('Duplicate card')
    return result


def _hole(values):
    return tuple(sorted(_cards(values, 2), key=CARDS.index))


@dataclass(frozen=True)
class PlayerState:
    seat: int
    stack: int
    bet_this_street: int
    hand_commit: int
    is_folded: bool
    is_all_in: bool


@dataclass(frozen=True)
class ActionEvent:
    index: int
    street: str
    seat: int | None
    kind: str
    amount: int
    paid: int
    to: int | None

    def raw(self):
        if self.kind == 'deal':
            return {'action': 'deal'}
        return {'seat': self.seat, 'action': self.kind, 'amount': self.amount}

    def to_engine_action(self):
        if self.kind not in ('fold', 'check', 'call', 'raise'):
            raise ValueError('Not a voluntary action')
        result = {'seat': self.seat, 'kind': self.kind}
        if self.kind == 'raise':
            result['to'] = self.to
        return result


class _Ledger:
    def __init__(self, dealer):
        self.dealer = dealer
        self.stacks = [STARTING_STACK]*6
        self.bets = [0]*6
        self.commits = [0]*6
        self.folded = [False]*6
        self.acted = [False]*6
        self.street = 0
        self.current_bet = 0
        self.min_raise = BIG_BLIND
        self.actor = (dealer+3)%6
        self.events = []
        self.awaiting_deal = False

    def active(self, seat):
        return not self.folded[seat] and self.stacks[seat] > 0

    def next_actor(self, after):
        for distance in range(1, 7):
            seat = (after+distance)%6
            if self.active(seat) and (not self.acted[seat] or self.bets[seat] != self.current_bet):
                return seat
        return None

    def bounds(self, seat):
        maximum = self.bets[seat]+self.stacks[seat]
        if not self.active(seat) or maximum <= self.current_bet:
            return None
        return (min(self.current_bet+self.min_raise, maximum), maximum)

    def snapshot(self, board, hole=None):
        actor = self.actor
        return PublicState(self.dealer, actor, STREETS[self.street], tuple(board[:BOARD_LENGTHS[self.street]]),
            hole, tuple(PlayerState(i,self.stacks[i],self.bets[i],self.commits[i],self.folded[i],self.stacks[i]==0)
                        for i in range(6)), tuple(self.events), self.current_bet, self.min_raise, self.bounds(actor))

    def consume(self, raw):
        if not isinstance(raw, dict) or set(raw)-{'seat','action','amount'}:
            raise ValueError('Unexpected history fields; hidden cards are forbidden')
        kind = raw.get('action')
        index = len(self.events)
        if kind == 'deal':
            if set(raw) != {'action'} or not self.awaiting_deal or self.street >= 3:
                raise ValueError('Unexpected street transition')
            self.street += 1
            self.bets = [0]*6
            self.current_bet = 0
            self.min_raise = BIG_BLIND
            self.acted = [False]*6
            self.actor = self.next_actor(self.dealer)
            if self.actor is None:
                raise ValueError('Terminal/all-in runout is not a decision observation')
            self.awaiting_deal = False
            event = ActionEvent(index,STREETS[self.street],None,kind,0,0,None)
            self.events.append(event)
            return event
        seat = _integer(raw.get('seat'),'history seat',5)
        amount = _integer(raw.get('amount'),'history amount',60000)
        if index < 2:
            expected = [('small_blind',(self.dealer+1)%6,50),('big_blind',(self.dealer+2)%6,100)][index]
            if (kind,seat,amount) != expected:
                raise ValueError('Expected fixed100BB six-seat blind posts')
            paid = amount
            self.stacks[seat] -= paid
            self.bets[seat] += paid
            self.commits[seat] += paid
            self.current_bet = max(self.bets)
        else:
            if self.awaiting_deal or seat != self.actor or not self.active(seat):
                raise ValueError('History actor or street mismatch')
            owed = self.current_bet-self.bets[seat]
            if kind == 'fold':
                if amount != 0 or owed == 0:
                    raise ValueError('Illegal fold')
                self.folded[seat] = True
                paid = 0
            elif kind in ('call','check'):
                paid = min(owed,self.stacks[seat])
                if (kind == 'check') != (owed == 0) or amount != paid:
                    raise ValueError('Call amounts are incremental; check must owe zero')
                self.acted[seat] = True
            elif kind == 'raise':
                bounds = self.bounds(seat)
                if bounds is None or not bounds[0] <= amount <= bounds[1]:
                    raise ValueError('Raise is a street total outside pinned-engine bounds')
                paid = amount-self.bets[seat]
                increment = amount-self.current_bet
                if increment >= self.min_raise:
                    self.min_raise = increment
                    self.acted = [False]*6
                self.current_bet = amount
                self.acted[seat] = True
            else:
                raise ValueError('Unknown history action')
            self.stacks[seat] -= paid
            self.bets[seat] += paid
            self.commits[seat] += paid
            if sum(not f for f in self.folded) <= 1:
                raise ValueError('Terminal fold history is not a decision observation')
            following = self.next_actor(seat)
            if following is None:
                if self.street == 3:
                    raise ValueError('Terminal showdown history is not a decision observation')
                self.awaiting_deal = True
            else:
                self.actor = following
        event = ActionEvent(index,STREETS[self.street],seat,kind,amount,paid,amount if kind=='raise' else None)
        self.events.append(event)
        return event


@dataclass(frozen=True)
class PublicState:
    dealer: int
    seat_to_act: int
    street: str
    board: tuple[str,...]
    hero_hole: tuple[str,str] | None
    players: tuple[PlayerState,...]
    events: tuple[ActionEvent,...]
    current_bet: int
    min_raise_increment: int
    raise_bounds: tuple[int,int] | None

    @classmethod
    def from_bridge(cls, state):
        return cls._from_observation(state, require_hole=True)

    @classmethod
    def from_public(cls, state):
        return cls._from_observation(state, require_hole=False)

    @classmethod
    def from_json(cls, state):
        if not isinstance(state, Mapping) or state.get('schema')!='poker-public-state-v1':
            raise ValueError('Unknown public-state schema')
        raw=dict(state);raw.pop('schema');increment=raw.pop('min_raise_increment',None)
        result=cls.from_public(raw)
        if type(increment) is not int or increment!=result.min_raise_increment:
            raise ValueError('Archived min-raise increment/history mismatch')
        return result

    @classmethod
    def _from_observation(cls, state, require_hole):
        fields={'terminal','your_cards','community_cards','dealer','seat_to_act','street','pot','your_stack',
                'your_bet_this_street','amount_owed','current_bet','min_raise_to','raise_bounds','can_check',
                'players','action_log'}
        if not isinstance(state,Mapping) or set(state)-fields or state.get('terminal') is not False:
            raise ValueError('Expected nonterminal public Bridge observation; unknown/hidden fields forbidden')
        dealer=_integer(state.get('dealer'),'dealer',5)
        board=_cards(state.get('community_cards'))
        if state.get('street') not in STREETS or len(board)!=BOARD_LENGTHS[STREETS.index(state['street'])]:
            raise ValueError('Board/street mismatch')
        hole=_cards(state['your_cards'],2) if state.get('your_cards') is not None else None
        if require_hole and hole is None:
            raise ValueError('Hero hole required')
        if hole and set(hole)&set(board):
            raise ValueError('Hero cards collide with public board')
        history=state.get('action_log')
        if not isinstance(history,list) or not 2<=len(history)<=515:
            raise ValueError('Complete explicit street-marked action history required')
        ledger=_Ledger(dealer)
        for event in history:ledger.consume(event)
        if ledger.awaiting_deal:
            raise ValueError('Missing deal marker')
        result=ledger.snapshot(board,hole)
        expected=result.to_observation()
        for key in fields-{'players','your_cards','action_log'}:
            if state.get(key)!=expected[key] or type(state.get(key)) is not type(expected[key]):
                raise ValueError('Observation/history mismatch: '+key)
        players=state.get('players')
        if not isinstance(players,list) or len(players)!=6:
            raise ValueError('Six public player states required')
        seen=set()
        for player in players:
            if not isinstance(player,dict) or set(player)-{'seat','stack','bet_this_street','hand_commit','is_folded','is_all_in'}:
                raise ValueError('Unexpected player fields; private cards forbidden')
            seat=_integer(player.get('seat'),'player seat',5)
            if seat in seen:raise ValueError('Repeated player seat')
            seen.add(seat);truth=result.players[seat]
            for key in ['stack','bet_this_street','is_folded','is_all_in']:
                if player.get(key)!=getattr(truth,key) or type(player.get(key)) is not type(getattr(truth,key)):
                    raise ValueError('Player/history mismatch: '+key)
            if 'hand_commit' in player and player['hand_commit']!=truth.hand_commit:
                raise ValueError('Player hand commitment mismatch')
        return result

    @property
    def pot(self):
        return sum(p.hand_commit for p in self.players)

    def public_only(self):
        return replace(self,hero_hole=None)

    def to_observation(self):
        p=self.players[self.seat_to_act]
        owed=self.current_bet-p.bet_this_street
        out=dict(terminal=False,community_cards=list(self.board),dealer=self.dealer,seat_to_act=self.seat_to_act,
            street=self.street,pot=self.pot,your_stack=p.stack,your_bet_this_street=p.bet_this_street,
            amount_owed=owed,current_bet=self.current_bet,min_raise_to=self.raise_bounds[0] if self.raise_bounds else None,
            raise_bounds=list(self.raise_bounds) if self.raise_bounds else None,can_check=owed==0,
            players=[dict(seat=p.seat,stack=p.stack,bet_this_street=p.bet_this_street,
                          is_folded=p.is_folded,is_all_in=p.is_all_in) for p in self.players],
            action_log=[e.raw() for e in self.events])
        if self.hero_hole is not None:out['your_cards']=list(self.hero_hole)
        return out

    def to_json(self):
        out=self.to_observation()
        for row,p in zip(out['players'],self.players):row['hand_commit']=p.hand_commit
        out['schema']='poker-public-state-v1'
        out['min_raise_increment']=self.min_raise_increment
        return out

    def canonical_suit_key(self, include_hole=True):
        """Identity up to one global suit renaming, never independent suit changes."""
        base=self.to_json()
        if not include_hole:base.pop('your_cards',None)
        candidates=[]
        for order in permutations(SUITS):
            mapping=dict(zip(SUITS,order));v=dict(base)
            v['community_cards']=[c[0]+mapping[c[1]] for c in self.board]
            if 'your_cards' in v:v['your_cards']=sorted(c[0]+mapping[c[1]] for c in v['your_cards'])
            candidates.append(json.dumps(v,sort_keys=True,separators=(',',':')))
        return min(candidates)


def observe_for_hole(state, seat, hole):
    """Observe a hypothetical acting hand using public data only.

    Passing an original observation strips its actual hero hole before adding
    the candidate. Seat must be the actor of this public decision/prefix.
    """
    public=state if isinstance(state,PublicState) else PublicState.from_public(state)
    if _integer(seat,'actor',5)!=public.seat_to_act:
        raise ValueError('Hypothetical hole must belong to the public actor')
    combo=_hole(hole)
    if set(combo)&set(public.board):raise ValueError('Hypothetical hole collides with public board')
    return replace(public,hero_hole=combo).to_observation()


def decision_prefixes(state):
    """Return (hole-free pre-action PublicState, observed ActionEvent) pairs."""
    public=state if isinstance(state,PublicState) else PublicState.from_bridge(state)
    ledger=_Ledger(public.dealer);result=[]
    for event in public.events:
        if event.kind in ('fold','check','call','raise'):
            result.append((ledger.snapshot(public.board),event))
        ledger.consume(event.raw())
    return tuple(result)


iter_decision_prefixes=decision_prefixes


@dataclass(frozen=True)
class Range:
    combos: tuple[tuple[str,str],...]
    weights: tuple[float,...]
    provenance: tuple[str,...] = ('unspecified_approximation',)

    def __post_init__(self):
        combos=tuple(_hole(h) for h in self.combos)
        weights=tuple(float(w) for w in self.weights)
        if not combos or len(set(combos))!=len(combos) or len(combos)!=len(weights):
            raise ValueError('Range must have unique combos and one weight per combo')
        if any(not math.isfinite(w) or w<0 for w in weights) or sum(weights)<=0:
            raise ValueError('Range weights must be finite nonnegative with positive mass')
        if isinstance(self.provenance,str) or not self.provenance or any(not isinstance(p,str) or not p for p in self.provenance):
            raise ValueError('Explicit range provenance required')
        total=sum(weights)
        if not math.isfinite(total):raise ValueError('Range total mass must be finite')
        object.__setattr__(self,'combos',combos)
        object.__setattr__(self,'weights',tuple(w/total for w in weights))
        object.__setattr__(self,'provenance',tuple(self.provenance))

    @classmethod
    def uniform(cls, board=()):
        """Uniform single-seat prior given PUBLIC cards, never actual hero cards."""
        blocked=set(_cards(board));hands=tuple(combinations((c for c in CARDS if c not in blocked),2))
        return cls(hands,(1.,)*len(hands),('exact_uniform_single_seat_prior_given_public_board',))

    @classmethod
    def from_weights(cls, combos, weights, provenance):
        return cls(tuple(combos),tuple(weights),(provenance,) if isinstance(provenance,str) else tuple(provenance))

    @property
    def hands(self):
        return self.combos

    def condition_on_blockers(self,cards):
        """An explicit marginal view for final action queries; not full own-range input.

        The range solver should instead receive both board-only ranges and mask
        colliding joint hands itself. Folded-seat card-removal is not modeled here.
        """
        blocked=set(_cards(cards));keep=[i for i,h in enumerate(self.combos) if not blocked.intersection(h)]
        return Range(tuple(self.combos[i] for i in keep),tuple(self.weights[i] for i in keep),
            self.provenance+('explicit_final_query_blocker_conditioning',))


def infer_range(state, seat, likelihood: Callable[[dict,ActionEvent],float], prior=None,
                *, model_provenance='approximate_blueprint_action_likelihood'):
    """Bayes reweight a marginal under an explicitly approximate public policy.

    No real hero cards enter construction or callbacks. The caller must reject
    unsupported off-tree action likelihoods rather than silently invent them.
    This is not an exact joint posterior: folded-seat removal and cross-seat
    action/carding correlations require a joint range model or solver treatment.
    """
    public=state if isinstance(state,PublicState) else PublicState.from_bridge(state)
    seat=_integer(seat,'range seat',5);initial=prior or Range.uniform(public.board)
    if any(set(h)&set(public.board) for h in initial.combos):raise ValueError('Prior collides with board')
    selected=[(prefix,event) for prefix,event in decision_prefixes(public) if event.seat==seat]
    weights=list(initial.weights)
    for prefix,event in selected:
        for i,hand in enumerate(initial.combos):
            value=float(likelihood(observe_for_hole(prefix,seat,hand),event))
            if not math.isfinite(value) or not 0<=value<=1:
                raise ValueError('Action likelihood must be finite probability')
            weights[i]*=value
        mass=sum(weights)
        if mass<=0:raise ValueError('Observed history has zero likelihood under candidate range/model')
        weights=[w/mass for w in weights]
    return Range(initial.combos,tuple(weights),initial.provenance+(model_provenance,
        'marginal_policy_factorization_not_exact_joint_posterior',
        'folded_player_card_removal_not_conditioned',f'own_actions_conditioned:{len(selected)}'))
