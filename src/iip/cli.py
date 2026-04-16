"""`iip` CLI — thin wrapper over the trainers, evaluator, and interactive play."""

from __future__ import annotations

import random
from pathlib import Path

import typer
from rich import print as rprint

from iip.agents.deep_cfr import DeepCFRAgent
from iip.agents.fixed_policy import FishAgent, StartingHandAgent, StrengthHandAgent
from iip.agents.random_agent import RandomAgent
from iip.engine.game import HULHE
from iip.engine.kuhn import Kuhn
from iip.metrics.exploitability import local_best_response
from iip.metrics.mbb import head_to_head_mbb
from iip.train.deepcfr_trainer import DeepCFRTrainer, DeepCFRTrainerConfig
from iip.train.game_adapter import HULHEAdapter, KuhnAdapter

app = typer.Typer(add_completion=False, help="Incomplete-Info-Problem CLI.")


@app.command()
def train(
    game: str = typer.Option("hulhe", help="hulhe | kuhn"),
    iters: int = typer.Option(5, help="Number of Deep CFR iterations."),
    traversals: int = typer.Option(2000, help="Traversals per iteration per player."),
    output: Path = typer.Option(Path("checkpoints/local/deepcfr.pt"), help="Where to save the strategy net."),
    seed: int = typer.Option(0),
    device: str = typer.Option("cpu"),
) -> None:
    """Train a Deep CFR agent end-to-end and save its strategy net."""
    if game == "hulhe":
        adapter = HULHEAdapter(game=HULHE())
    elif game == "kuhn":
        adapter = KuhnAdapter(game=Kuhn())
    else:
        raise typer.BadParameter(f"Unknown game {game!r}")

    cfg = DeepCFRTrainerConfig(
        traversals_per_iteration=traversals,
        advantage_train_steps=500,
        strategy_train_steps=1000,
        batch_size=256,
        device=device,
    )
    trainer = DeepCFRTrainer(adapter=adapter, config=cfg, seed=seed)
    trainer.iterate(n_iters=iters)
    agent = trainer.finalize_strategy()
    agent.save(output)
    rprint(f"[green]Saved Deep CFR strategy net -> {output}[/green]")


@app.command()
def eval(
    game: str = typer.Option("hulhe"),
    checkpoint: Path = typer.Option(..., exists=True, help="Path to a Deep CFR strategy net."),
    hands: int = typer.Option(2000),
    opponents: str = typer.Option("fish,strength,random",
                                  help="Comma-separated subset of fish,strength,starting,random."),
    seed: int = typer.Option(0),
    device: str = typer.Option("cpu"),
    with_lbr: bool = typer.Option(False, help="Also compute LBR exploitability proxy."),
) -> None:
    """Evaluate a checkpoint head-to-head vs baselines; optionally report LBR exploitability."""
    if game != "hulhe":
        raise typer.BadParameter("Only HULHE evaluation is supported here.")
    gm = HULHE()
    hero = DeepCFRAgent.load(checkpoint, device=device)
    bank = {
        "fish": FishAgent(),
        "strength": StrengthHandAgent(),
        "starting": StartingHandAgent(),
        "random": RandomAgent(rng=random.Random(seed)),
    }
    for key in opponents.split(","):
        key = key.strip()
        if key not in bank:
            rprint(f"[red]Unknown opponent {key!r}, skipping.[/red]")
            continue
        r = head_to_head_mbb(gm, hero, bank[key], n_hands=hands, seed=seed)
        rprint(str(r))
    if with_lbr:
        lbr = local_best_response(gm, hero, n_hands=min(hands, 1000), seed=seed)
        rprint(f"[bold]LBR exploitability:[/bold] {lbr:.2f} mbb/h (higher = more exploitable)")


@app.command()
def play(
    checkpoint: Path = typer.Option(..., exists=True),
    hands: int = typer.Option(5),
    device: str = typer.Option("cpu"),
) -> None:
    """Interactive CLI — play N hands against the bot. Useful as a quick sanity harness; the
    Streamlit app is the real UI."""
    gm = HULHE()
    bot = DeepCFRAgent.load(checkpoint, device=device)
    for h in range(hands):
        rprint(f"\n[bold]--- Hand {h + 1} ---[/bold]")
        hero_seat = h % 2
        state = gm.new_hand(rng=random.Random())
        while not state.terminal:
            rprint(f"Street={state.street.name} pot={state.pot} stacks={state.stacks} contrib={state.contributions}")
            if state.to_act == hero_seat:
                rprint(f"Your hole cards: {state.hole_cards[hero_seat]}  Board: {state.board}")
                legal = gm.legal_actions(state)
                rprint(f"Legal: {[a.name for a in legal]}")
                choice = typer.prompt("Action (FOLD/CHECK_CALL/BET_RAISE)").upper()
                act = next(a for a in legal if a.name == choice)
                gm.step(state, act)
            else:
                act = bot.act(gm, state, state.to_act)
                rprint(f"[cyan]Bot {state.to_act}[/cyan]: {act.name}")
                gm.step(state, act)
        payoff = gm.payoffs(state)
        rprint(f"[green]Payoff[/green] — hero: {payoff[hero_seat]}, bot: {payoff[1 - hero_seat]}")


if __name__ == "__main__":
    app()
