# Tech Docs Workshop: Document a Hotel Cancellation Model

This is a practice repo: a finished machine learning project with almost no documentation. Your team uses Copilot to document it, and checks every claim it makes. Afterward, you can use the same steps on your own repo.

## What You'll Learn

By the end of the workshop, you should be able to:

- Explore an unfamiliar codebase and identify which files support important documentation claims.
- Write useful docstrings and a README with AI assistance.
- Check whether commands, dependencies, and reported results are supported by the project.
- Recognize when files disagree and decide what evidence is trustworthy.
- Apply the Explore, Plan, Build, Check workflow to another project.

You should be comfortable reading basic Python and navigating GitHub. You do not need prior machine-learning experience.

## How This Works

- One person drives and shares their screen. Everyone else reads each change and calls out problems. Switch drivers between parts.
- Do Parts 1, 2, and 3 in order. Each part ends with a CHECK. When the CHECK passes, move on.
- Finished early? Try the Challenge.
- Stuck? Ask your coach.
- The checks are useful guardrails, not proof that every sentence in your documentation is correct. They test specific things; you still need to review claims and sources yourself.

## Where Things Are

- **Copilot Chat:** the chat icon at the top of the window. Pick **Agent** in the mode menu under the chat box, if it is available in your environment.
- **Terminal:** the panel at the bottom. If it's closed, open the menu at the top left and pick Terminal, then New Terminal.
- **Keep or Undo:** after Copilot edits a file, read the change. Then click **Keep** to accept it or **Undo** to reject it.
- **Context:** type `#` in the chat box to point Copilot at something, like `#codebase` for the repo or `#file:src/train.py` for one file. Context references help Copilot locate information; they do not guarantee that every relevant file will be considered.
- **checks/:** has the functions that check your work. You don't need to open it. Please don't ask Copilot to change it.

## Setup

Run this in the terminal:

```bash
python checks/check_docs.py
```

**Success check:** you see "You're ready."

This confirms that the project environment is ready for the workshop. It does not test your Copilot sign-in or guarantee that Agent mode is available. If Copilot Chat is unavailable, tell your coach so you can use the fallback instructions.

---
## Part 1: Docstrings

**Goal:** Add docstrings that describe what the functions actually do, without changing the code.

1. Paste this into Copilot Chat:

   > Read #file:src/features.py, #file:src/train.py, and #file:src/evaluate.py. Add a Google-style docstring to every function: a one-line summary, then Args and Returns where they apply. Describe what the code actually does. Don't change any code. If an existing docstring conflicts with the implementation, point out the mismatch and correct the docstring.

2. Read each change. Did Copilot flag anything about the docstrings that were already there? Compare the description with the implementation, especially `fill_missing_agent()`. Then click **Keep** only after reviewing the changes.
3. Run CHECK 1:

   ```bash
   python checks/check_docs.py 1
   ```

**Success check:** CHECK 1 says "Nice work." This check looks for docstrings on the expected functions and catches a known mismatch in this exercise. It does not prove that every docstring is complete or semantically correct. If it shows warnings, inspect the implementation, fix the documentation, and run it again.

---
## Part 2: README

1. **Explore.** Paste this. Copilot should answer without changing any files.

   > Read #codebase. Summarize what this project does and trace the workflow from the input data to the model evaluation. Identify the files that create outputs. List every reported performance number and the file each number comes from. Note any conflicting or potentially outdated results, and distinguish what you can verify from what you cannot. Don't edit any files yet.

2. **Plan.** Paste the strong prompt from the slides, below. First, add one useful finding from Explore to Context. If you found conflicting numbers, include the filenames and values rather than silently choosing one.

   > **Goal:** Write README.md for this project. The reader is a recruiter or Challenge Advisor who won't run the code.
   >
   > **Context:** This finished ML project predicts hotel booking cancellations. Facts about the data are in data/README.md. *(Add what you learned from Explore, including any conflicting results.)*
   >
   > **Rules:** Use these sections: Overview, Data, Setup, Usage, Results, Limitations, Team. Use only facts you can find in this repo, and write TODO if you can't find one. Don't invent commands, flags, or numbers. Put commands in bash code blocks. Only edit README.md. If sources disagree, call out the disagreement and identify the files; do not silently pick a value.
   >
   > **Success check:** Every command is either tested or clearly marked for manual verification, and every number has a named source. Important results are independently reproduced from the current code.
   >
   > Show us an outline first, with the file each section's facts come from. For every result, say whether its source is current code output or a saved report. Don't write the README until we approve.

3. Read Copilot's outline. Reply "Approved," or tell it what to change. Check that its proposed sources are appropriate and that conflicting sources are not being treated as equally current.
4. **Build.** Let Copilot write the README. Read every line, then click **Keep** only after checking it.
5. Replace the TODO in the Team section with your names.
6. **Check.** Run CHECKS 2, 3, and 4. CHECK 4 runs supported Python and pip commands in your README. Commands it cannot safely interpret are marked for manual verification; do not treat a skipped command as tested.

   ```bash
   python checks/check_docs.py 2
   python checks/check_docs.py 3
   python checks/check_docs.py 4
   ```

**Success check:** all three say "Nice work." If CHECK 3 asks you to fix `requirements.txt`, inspect the imports and list direct dependencies explicitly. A package working only because another package happens to install it is not a reliable dependency declaration.

---
## Part 3: Find It Yourself

1. Run CHECK 5:

   ```bash
   python checks/check_docs.py 5
   ```

2. If it shows a warning, find out why. Use Copilot to look around, but you decide what's true and how to fix it. Asking Copilot to "fix the README" won't get you there.
3. Compare the results in `results/metrics.json` with the results produced by the current training and evaluation code. The saved file may describe a different run. A file is evidence, not automatically the source of truth.
4. Run CHECK 5 again after you have resolved the conflict in your README and dealt with the stale report.

**Success check:** CHECK 5 says "PASS: the checked Results metrics and test-set count match." In this exercise, CHECK 5 checks the overall accuracy, precision, recall, and held-out test-set count against a fresh run, and flags the known saved-metrics conflict. It does not validate every number or claim elsewhere in the README.

Then write down what you found:
- What was wrong:
- Which files disagreed:
- How you independently established the current result:
- Why might Copilot have missed it?
- What would you change in your documentation workflow next time?

For reference, the data pipeline starts with 20,000 rows in the supplied CSV, removes duplicate bookings, and then uses a stratified 80/20 split. The cleaned row count and held-out test-set size are determined by the code; verify them rather than assuming that the original CSV row count is the model's test-set size.

---
## Challenge: Where Does It Struggle?

A strong README is honest about where a project falls short. This time, write your own prompts using Explore, Plan, Build, Check.

1. Find recall for City Hotel and Resort Hotel bookings separately. Nothing in the repo prints these yet. Ask Copilot to write a short script that does, and read it before you run it.
2. Add both numbers to the README's Limitations section. Label each one, like `City Hotel recall: 0.00`, or put them in a table.
3. In a sentence or two, say what those numbers mean for someone using this model. Do not claim more than the numbers support.
4. Only claim what you can show. The check flags claims like "production-ready."
5. Run the Challenge check:

   ```bash
   python checks/check_docs.py challenge
   ```

**Success check:** every item says PASS. The hotel-specific recall exercise is specific to this ML project; the broader habit of documenting limitations with evidence applies to other kinds of projects too.

---
## More to Try

- **Team rules:** create `.github/copilot-instructions.md` with your docstring style and documentation rules. Copilot can use these instructions as context for supported requests, but don't assume they are applied in every tool or mode. Then ask Copilot to document a new function and compare.
- **Your project:** open your Challenge Project repo and start with the Explore prompt from Part 2. Identify the most relevant and reliable sources before asking Copilot to draft documentation.
