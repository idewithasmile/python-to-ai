# Study Guide

The curriculum is designed for steady, deliberate practice. The goal is not to finish quickly; the goal is to build durable understanding and a portfolio of work that you can defend.

## Weekly Rhythm

Plan for 8–12 focused hours per week:

| Phase | Time | Activity |
| :--- | :--- | :--- |
| Plan | 30 minutes | Read the week's outcomes, inspect the dataset or starter code, and write a short plan. |
| Learn | 2–3 hours | Read the theory, derive the key equations, and take notes in your own words. |
| Build | 4–6 hours | Complete the guided notebook and at least one exercise without copying a solution. |
| Explain | 1 hour | Write a lab note describing what worked, what failed, and why. |
| Review | 30 minutes | Re-run the notebook from a clean kernel, check results, and update the roadmap. |

If a week is disrupted, protect the milestone. Reduce optional exercises, use smaller data, or move the review checkpoint. Do not skip the project review.

## Notebook Standards

Use one notebook per lesson and keep it reproducible:

1. **Title and objective:** state the question and the expected output.
2. **Setup:** imports, seeds, paths, and dependency versions.
3. **Theory:** concise markdown with equations or diagrams where useful.
4. **Implementation:** small, readable code cells with meaningful names.
5. **Experiments:** one variable changed at a time, with a short interpretation.
6. **Exercises:** at least one variation that is not solved in the lesson.
7. **Results:** metrics, plots, and a comparison to a baseline.
8. **Reflection:** limitations, failure cases, and one thing to revisit.

Before committing:

- Restart the kernel and run all cells from top to bottom.
- Remove credentials, absolute personal paths, and irrelevant outputs.
- Confirm that all random seeds are set.
- Keep plots lightweight and save important figures as files.
- Never commit datasets larger than 50 MB or model checkpoints.

## Project Rubric

Milestone projects are reviewed against five criteria:

| Criterion | Weight | What good looks like |
| :--- | :---: | :--- |
| Problem and data | 20% | Clear question, appropriate dataset, documented license, honest limitations. |
| Method | 25% | Correct implementation, sensible baseline, justified design choices. |
| Evaluation | 25% | Leakage-free splits, relevant metrics, uncertainty, error analysis. |
| Reproducibility | 15% | Seeds, versions, commands, configuration, and environment documented. |
| Communication | 15% | Readable code, clear README, useful figures, concise reflection. |

A project that reports only a single accuracy number is incomplete. A strong project explains where the model fails and what would be tried next.

## Experiment Checklist

For every non-trivial experiment, record:

- Hypothesis being tested.
- Baseline and proposed change.
- Dataset version and split.
- Random seeds.
- Hyperparameters and search space.
- Hardware and runtime.
- Metrics with confidence intervals or repeated runs when practical.
- Qualitative examples, especially failures.
- Decision: keep, revise, or discard the idea.

Use the experiment log template in [`templates/experiment-log.md`](../templates/experiment-log.md).

## Git Workflow

Commit at meaningful checkpoints, not once at the end of a module. Small commits make debugging and portfolio review easier.

Suggested commit format:

```text
type(scope): short imperative summary
```

Common types:

- `feat(m03)`: new lesson, notebook, or project capability.
- `fix(m05)`: correction to code, math, or instructions.
- `docs(m08)`: documentation or explanation improvement.
- `test(m01)`: tests or validation.
- `chore`: tooling, dependencies, or repository maintenance.

Examples:

```text
feat(m01): add data-structure exercises
fix(m02): correct gradient descent notation
docs(m04): add model card template notes
```

Before committing:

```powershell
git status
git diff --check
python -m pytest
python -m ruff check .
```

## Progress Reviews

### Weekly Review

- Re-run the week's notebook from a clean kernel.
- Check that the output matches the stated objective.
- Update the weekly checkbox in the master roadmap.
- Write one paragraph about the most important thing learned.

### Module Review

- Complete the milestone project and its README.
- Verify the rubric in this guide.
- Update the module checkbox and target date if needed.
- Link the project in your portfolio or personal site.

### Quarterly Review

- Compare actual progress with the roadmap.
- Identify topics that need spaced repetition.
- Archive abandoned experiments.
- Adjust the next quarter without removing milestones.

## Asking Good Questions

A useful question includes:

- The exact goal.
- A minimal reproducible example or notebook cell.
- The expected result and the actual result.
- The full error message and traceback.
- What has already been tried.
- The environment and package versions.

Search the documentation first, then open an issue with the repository's learning-question template.

## Portfolio Strategy

Each milestone should produce:

- A project README with problem, data, method, results, and limitations.
- A short technical note or blog post.
- A reproducible command or script.
- A figure or demo that communicates the result quickly.
- A reflection on what you would do differently.

By the end of the curriculum, the portfolio should tell a coherent story: strong Python foundations, rigorous evaluation, modern deep learning, and responsible deployment.

## Common Failure Modes

- **Tutorial copying:** completing notebooks without writing independent code.
- **Metric worship:** optimizing one number without error analysis.
- **Data leakage:** tuning on the test set or preprocessing before splitting.
- **Hidden state:** notebooks that only work after cells are run out of order.
- **Undocumented experiments:** results that cannot be reproduced.
- **Scope creep:** expanding a milestone until it cannot be finished.
- **Skipping the math:** using a model without understanding its assumptions.

## Ethical and Professional Practice

- Use datasets with clear licenses and attribution.
- Protect personal and sensitive information.
- Document bias, consent, privacy, and misuse risks.
- Do not present generated media as real.
- Give credit to papers, libraries, and external resources.
- Report negative results honestly.
- Ask for help early, and share what you learn.
