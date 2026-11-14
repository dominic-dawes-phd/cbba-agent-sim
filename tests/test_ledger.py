"""The replicated ledger: merge algebra, local changes, leases, and convergence."""

import itertools
import random

from sim.ledger import Ledger, LedgerRow, decode_rows, encode_rows, merged


def some_rows():
    """Rows that differ in every field the merge looks at."""
    return [
        LedgerRow("unclaimed", None, 0.0, 0, None),
        LedgerRow("claimed", "arm_a", 0.5, 1, 40),
        LedgerRow("claimed", "arm_b", 0.5, 1, 40),
        LedgerRow("claimed", "arm_b", 0.9, 1, 40),
        LedgerRow("in_progress", "arm_a", 0.5, 2, 60),
        LedgerRow("in_progress", "arm_a", 0.5, 2, 90),
        LedgerRow("unclaimed", None, 0.0, 3, None),
        LedgerRow("done", "arm_b", 0.9, 2, None),
    ]


def test_merging_is_commutative_associative_and_idempotent():
    rows = some_rows()
    for a, b in itertools.product(rows, repeat=2):
        assert merged(a, b) == merged(b, a)
        assert merged(a, a) == a
    for a, b, c in itertools.product(rows, repeat=3):
        assert merged(merged(a, b), c) == merged(a, merged(b, c))


def test_done_always_wins_and_a_release_beats_the_claim_it_replaces():
    done = LedgerRow("done", "arm_a", 0.1, 1, None)
    late_claim = LedgerRow("in_progress", "arm_b", 0.9, 7, 99)
    assert merged(done, late_claim) == done
    ledger = Ledger("arm_a", lease_ticks=30)
    ledger.claim("t_1", 0.7, tick=10)
    claimed = ledger.row("t_1")
    ledger.release("t_1")
    assert merged(claimed, ledger.row("t_1")).status == "unclaimed"
    ledger.done("t_1")
    ledger.release("t_1")
    assert ledger.row("t_1").status == "done"


def test_a_claim_lapses_without_renewal_and_a_renewal_wins_without_a_new_version():
    ledger = Ledger("arm_a", lease_ticks=30)
    ledger.claim("t_1", 0.7, tick=10)
    assert not ledger.lapsed("t_1", 40) and ledger.lapsed("t_1", 41)
    before = ledger.row("t_1")
    ledger.renew("t_1", tick=35)
    after = ledger.row("t_1")
    assert after.version == before.version and after.lease_expiry == 65
    assert merged(before, after) == after


def test_copies_converge_after_gossip_in_any_order():
    agents = [Ledger(agent_id, lease_ticks=30) for agent_id in ("arm_a", "arm_b", "arm_c")]
    agents[0].claim("t_1", 0.4, tick=0)
    agents[1].claim("t_1", 0.8, tick=0)  # a race: the higher bid wins everywhere
    agents[2].add("t_2")
    agents[0].done("t_3")
    changes = [ledger.take_changes() for ledger in agents]
    messages = [(index, encode_rows(change)) for index, change in enumerate(changes)]
    for order in itertools.permutations(messages):
        copies = [Ledger(ledger.agent_id, 30) for ledger in agents]
        for copy, ledger in zip(copies, agents, strict=True):
            copy.merge(ledger.rows)
        for sender, payload in order:
            for index, copy in enumerate(copies):
                if index != sender:
                    copy.merge(decode_rows(payload))
        assert copies[0].rows == copies[1].rows == copies[2].rows
        assert copies[0].row("t_1").owner == "arm_b" and copies[0].row("t_3").status == "done"


def test_a_partition_heals_by_anti_entropy():
    rng = random.Random(0)
    left, right = Ledger("arm_a", 30), Ledger("arm_b", 30)
    for tick in range(20):
        side = left if rng.random() < 0.5 else right
        task = f"t_{rng.randrange(4)}"
        side.claim(task, rng.random(), tick)
        if rng.random() < 0.3:
            side.done(task)
    left.merge(right.rows)
    right.merge(left.rows)
    assert left.rows == right.rows
