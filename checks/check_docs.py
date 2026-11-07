"""
Checks for the Tech Docs workshop.

Run them from the terminal after each part:

    python checks/check_docs.py        CHECK 0: setup
    python checks/check_docs.py 1      CHECK 1: docstrings (Part 1)
    python checks/check_docs.py 2      CHECK 2: README sections (Part 2)
    python checks/check_docs.py 3      CHECK 3: requirements (Part 2)
    python checks/check_docs.py 4      CHECK 4: README commands (Part 2)
    python checks/check_docs.py 5      CHECK 5: README numbers (Part 3)
    python checks/check_docs.py challenge  Challenge

They don't fix anything. They tell you whether the docs hold up.
Please don't ask Copilot to edit this file. The checks only help if
they stay the same for every team.
"""

import ast
import hashlib
import importlib.metadata
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
README = ROOT / "README.md"
REQUIREMENTS = ROOT / "requirements.txt"
OLD_RESULTS = ROOT / "results" / "metrics.json"

LOCAL_MODULES = {"features", "train", "evaluate"}
IMPORT_TO_PACKAGE = {"sklearn": "scikit-learn"}
METRICS = ["accuracy", "precision", "recall"]
TOLERANCE = 0.01

README_SECTIONS = {
    "Overview": ["overview", "about", "introduction", "summary", "description"],
    "Data": ["data"],
    "Setup": ["setup", "set up", "install", "getting started", "requirements"],
    "Usage": ["usage", "how to run", "running", "quick start", "quickstart"],
    "Results": ["results", "performance", "evaluation", "metrics"],
    "Limitations": ["limitation", "known issues", "caveats"],
    "Team": ["team", "contributors", "authors"],
}
OVERCLAIMS = [
    "production-ready", "production ready", "state-of-the-art", "state of the art",
    "highly accurate", "flawless", "cutting-edge", "best-in-class", "industry-leading",
]

# Fingerprints of the original code, with docstrings and type hints removed.
# Adding docstrings is fine. Changing what a function does is not.
CODE_FINGERPRINTS = {
    "features.py": {
        "drop_duplicate_bookings": "77e20b6976",
        "fill_missing_agent": "92c5bdc121",
        "fix_lead_time_units": "3bf8a7eb3a",
        "drop_leaky_columns": "b891f991c5",
        "add_booking_features": "ac3569cadd",
        "clean_bookings": "3f9d65f031",
    },
    "train.py": {
        "load_data": "584302a64c",
        "build_pipeline": "50cd768bf4",
        "split_data": "48a101a483",
    },
    "evaluate.py": {
        "compute_metrics": "f6f18648f8",
    },
}


def _line():
    print("-" * 60)


def _header(title):
    _line()
    print(title)
    _line()


# ----------------------------------------------------------------------
# Helpers: code
# ----------------------------------------------------------------------
def _functions(path):
    tree = ast.parse(path.read_text())
    return [n for n in tree.body if isinstance(n, ast.FunctionDef)]


def _fingerprint(func):
    node = ast.parse(ast.unparse(func)).body[0]
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:] or [ast.Pass()]
    node.body = body
    for arg in node.args.args + node.args.kwonlyargs:
        arg.annotation = None
    node.returns = None
    return hashlib.md5(ast.unparse(node).encode()).hexdigest()[:10]


def _returns_value(func):
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node.value is not None:
            return True
    return False


def _true_results():
    """Train and score the model exactly as the code in src/ does today."""
    sys.path.insert(0, str(SRC))
    import evaluate
    import features
    import train

    df = features.clean_bookings(train.load_data())
    X_train, X_test, y_train, y_test = train.split_data(df)
    model = train.build_pipeline().fit(X_train, y_train)
    results = {
        "overall": evaluate.compute_metrics(model, X_test, y_test),
        "n_test_bookings": len(X_test),
    }
    for hotel in ["City Hotel", "Resort Hotel"]:
        rows = X_test["hotel"] == hotel
        results[hotel] = evaluate.compute_metrics(model, X_test[rows], y_test[rows])
    return results


# ----------------------------------------------------------------------
# Helpers: markdown
# ----------------------------------------------------------------------
def _headings(text):
    return [(len(m.group(1)), m.group(2).strip()) for m in re.finditer(r"^(#{1,6})\s+(.+)$", text, re.M)]


def _find_sections(text, wanted):
    found = {}
    titles = [t.lower() for _, t in _headings(text)]
    for name, keys in wanted.items():
        found[name] = any(any(k in t for k in keys) for t in titles)
    return found


def _section_text(text, keys):
    """Text under the first heading that matches any key, up to the next heading of the same level."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m and any(k in m.group(2).lower() for k in keys):
            level = len(m.group(1))
            out = []
            for nxt in lines[i + 1:]:
                n = re.match(r"^(#{1,6})\s+", nxt)
                if n and len(n.group(1)) <= level:
                    break
                out.append(nxt)
            return "\n".join(out)
    return ""


def _score_in(text):
    """First number in text that reads as a score, like 0.79, 79%, or 79.1. Skips counts and thresholds."""
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?\s*%?", text):
        token = m.group().replace(" ", "")
        before, after = text[:m.start()].lower(), text[m.end():].lower()
        is_threshold = re.search(r"threshold\s*(of|=|at)?\s*$", before) or re.match(r"\s*threshold", after)
        if "," in token or is_threshold:
            continue
        percent = token.endswith("%")
        value = float(token.rstrip("%"))
        if value > 100:
            continue
        return value / 100 if percent or value > 1 else value
    return None


def _hotel(label):
    label = label.lower()
    if "city" in label:
        return "City Hotel"
    if "resort" in label:
        return "Resort Hotel"
    return "overall"


def _metric(label):
    label = label.lower()
    for name in METRICS:
        if name in label:
            return name
    return None


def _extract_numbers(text):
    """Find labeled metrics in tables and in plain lines. Returns (hotel, metric, value) in order."""
    found = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            header, rows = block[0], [r for r in block[1:] if not all(set(c) <= set("-: ") for c in r)]
            for row in rows:
                for col, cell in enumerate(row[1:], start=1):
                    value = _score_in(cell)
                    if value is None:
                        continue
                    label = row[0] + " " + (header[col] if col < len(header) else "")
                    metric = _metric(label)
                    if metric:
                        found.append((_hotel(label), metric, value))
            continue
        line = lines[i]
        start, hotel = 0, "overall"
        for m in re.finditer(r"\b(accuracy|precision|recall)\b", line, re.I):
            named = _hotel(line[start:m.end()])
            if named == "overall" and start == 0:
                named = _hotel(line)
            hotel = named if named != "overall" else hotel
            value = _score_in(line[m.end():m.end() + 60])
            if value is not None:
                found.append((hotel, m.group(1).lower(), value))
            start = m.end()
        i += 1
    return found


def _first(found, hotel, metric):
    for h, m, v in found:
        if h == hotel and m == metric:
            return v
    return None


# ----------------------------------------------------------------------
# CHECK 0: Setup
# ----------------------------------------------------------------------
def check_setup():
    _header("CHECK 0: Is your Codespace ready?")
    ok = True
    for pkg in ["pandas", "scikit-learn"]:
        try:
            importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            print(f"WARNING: {pkg} isn't installed. Run: pip install -r requirements.txt")
            ok = False
    for path in ["src/features.py", "src/train.py", "src/evaluate.py", "data/hotel_bookings_sample.csv"]:
        if not (ROOT / path).exists():
            print(f"WARNING: {path} is missing. Make a fresh copy of the template.")
            ok = False
    if ok:
        print("PASS: everything is installed and in place.")
        print()
        print("You're ready. Open Copilot Chat and start Part 1 in WORKSHOP.md.")


# ----------------------------------------------------------------------
# CHECK 1: Docstrings
# ----------------------------------------------------------------------
def check_docstrings():
    _header("CHECK 1: Does every function have a docstring that matches its code?")
    ok = True
    total = documented = 0
    for path in sorted(SRC.glob("*.py")):
        for func in _functions(path):
            if func.name.startswith("_"):
                continue
            total += 1
            where = f"{path.name}: {func.name}()"
            doc = ast.get_docstring(func)
            if not doc:
                print(f"WARNING: {where} has no docstring.")
                ok = False
                continue
            documented += 1
            has_params = bool(func.args.args or func.args.kwonlyargs)
            if has_params and "Args:" not in doc:
                print(f"WARNING: {where} needs an Args: section.")
                ok = False
            if _returns_value(func) and "Returns:" not in doc:
                print(f"WARNING: {where} needs a Returns: section.")
                ok = False

    print(f"{documented} of {total} public functions have a docstring.")

    changed = []
    for filename, expected in CODE_FINGERPRINTS.items():
        funcs = {f.name: f for f in _functions(SRC / filename)}
        for name, fp in expected.items():
            if name not in funcs or _fingerprint(funcs[name]) != fp:
                changed.append(f"{filename}: {name}()")
    if changed:
        print()
        print("WARNING: the code inside these functions changed:")
        for item in changed:
            print(f"  {item}")
        print("Part 1 is docstrings only. Undo the code changes and keep the docstrings.")
        print("Did Copilot change code to match an old docstring? That's a common mistake.")
        ok = False

    funcs = {f.name: f for f in _functions(SRC / "features.py")}
    doc = ast.get_docstring(funcs.get("fill_missing_agent")) if "fill_missing_agent" in funcs else ""
    if doc and re.search(r"\bdrop", doc, re.I) and not re.search(r"(not|n't|never|without|no)\W+(\w+\W+){0,3}drop", doc, re.I):
        print()
        print("WARNING: at least one docstring describes something its function doesn't do.")
        print("Read each docstring next to its code. Trust the code, not the old docstring.")
        ok = False

    print()
    print("Nice work. Move on to Part 2." if ok else "Fix the warnings, then run CHECK 1 again.")


# ----------------------------------------------------------------------
# CHECK 2: README sections
# ----------------------------------------------------------------------
def check_readme_sections():
    _header("CHECK 2: Does the README cover what a reader needs?")
    text = README.read_text()
    ok = True
    for name, present in _find_sections(text, README_SECTIONS).items():
        if present:
            print(f"PASS: {name}")
        else:
            print(f"WARNING: no {name} section.")
            ok = False
    if not re.search(r"antonio|data in brief", text, re.I):
        print("WARNING: the README doesn't credit where the data came from.")
        ok = False
    if not re.search(r"cc[ -]?by", text, re.I):
        print("WARNING: the README doesn't name the data's license.")
        ok = False
    leftovers = re.findall(r"TODO|\[your[^\]]*\]|your-username|lorem ipsum", text, re.I)
    if leftovers:
        print(f"WARNING: the README still has {len(leftovers)} placeholder(s), like TODO. Fill them in.")
        ok = False
    print()
    print("Nice work. Run CHECK 3." if ok else "Fix the warnings, then run CHECK 2 again.")


# ----------------------------------------------------------------------
# CHECK 3: Requirements
# ----------------------------------------------------------------------
def _requirement_names():
    names = []
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            names.append(re.split(r"[<>=!~\[; ]", line)[0].lower().replace("_", "-"))
    return names


def check_requirements():
    _header("CHECK 3: Does requirements.txt list everything the code needs?")
    ok = True
    listed = set(_requirement_names())
    needed = {}
    for path in sorted(SRC.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module.split(".")[0]]
            for mod in mods:
                if mod in LOCAL_MODULES or mod in sys.stdlib_module_names:
                    continue
                needed.setdefault(IMPORT_TO_PACKAGE.get(mod, mod).lower(), set()).add(path.name)

    for pkg, files in sorted(needed.items()):
        if pkg in listed:
            print(f"PASS: {pkg}")
        else:
            names = sorted(files)
            who = names[0] + " imports" if len(names) == 1 else " and ".join(names) + " import"
            print(f"WARNING: {who} {pkg}, but requirements.txt doesn't list it.")
            print("  It may run here because another package installs it. Don't count on that.")
            ok = False

    for pkg in sorted(listed):
        try:
            importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            print(f"WARNING: requirements.txt lists {pkg}, but it isn't installed.")
            print("  Does that package exist? Look it up on pypi.org before you keep it.")
            ok = False
        else:
            if pkg not in needed:
                print(f"NOTE: requirements.txt lists {pkg}, but nothing in src/ imports it. Is it needed?")
    print()
    print("Nice work. Run CHECK 4." if ok else "Fix the warnings, then run CHECK 3 again.")


# ----------------------------------------------------------------------
# CHECK 4: README commands
# ----------------------------------------------------------------------
def _readme_commands(text):
    commands = []
    for block in re.findall(r"```[ \t]*(?:bash|sh|shell|console|zsh)?[ \t]*\n(.*?)```", text, re.S):
        for raw in block.splitlines():
            raw = raw.strip()
            if raw.startswith("$ "):
                raw = raw[2:]
            if not raw or raw.startswith("#"):
                continue
            commands.extend(part.strip() for part in raw.split("&&") if part.strip())
    return commands


def _runnable(cmd):
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return None
    if not parts:
        return None
    if parts[0] in ("pip", "pip3") and parts[1:3] == ["install", "-r"]:
        return [sys.executable, "-m", "pip", "install", "-q"] + parts[2:]
    if parts[0] in ("python", "python3") and parts[1:4] == ["-m", "pip", "install"]:
        return [sys.executable, "-m", "pip", "install", "-q"] + parts[4:]
    if parts[0] in ("python", "python3") and len(parts) > 1:
        return [sys.executable] + parts[1:]
    return None


def check_commands():
    _header("CHECK 4: Do the README's commands actually run?")
    commands = _readme_commands(README.read_text())
    if not commands:
        print("WARNING: no commands found. Put Setup and Usage commands in ```bash code blocks.")
        print()
        print("Fix the warning, then run CHECK 4 again.")
        return

    ok = True
    models = ROOT / "models"
    temp_root = Path(tempfile.mkdtemp(prefix="tech-docs-check-"))
    backup = temp_root / "models-backup"
    had_models = models.exists()
    if had_models:
        shutil.move(str(models), str(backup))

    print("Running supported commands from a clean model-output state.")
    print("Commands outside the supported Python/pip subset require manual verification.")
    print()
    try:
        for cmd in commands:
            argv = _runnable(cmd)
            if argv is None:
                print(f"WARNING: NOT RUN: {cmd}")
                print("  Check this command manually; CHECK 4 cannot verify it.")
                ok = False
                continue
            try:
                result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired:
                print(f"WARNING: {cmd}  (took more than 5 minutes)")
                ok = False
                continue
            if result.returncode == 0:
                print(f"PASS: {cmd}")
            else:
                ok = False
                print(f"WARNING: {cmd}  (failed)")
                err = (result.stderr or result.stdout).strip().splitlines()
                for line in err[-3:]:
                    print(f"    {line}")
    finally:
        # Remove artifacts created by README commands, then restore the user's
        # original models directory exactly as it was before the check.
        if models.exists():
            shutil.rmtree(models) if models.is_dir() else models.unlink()
        if had_models and backup.exists():
            shutil.move(str(backup), str(models))
        shutil.rmtree(temp_root, ignore_errors=True)

    joined = " ".join(commands)
    for need, label in [("requirements.txt", "install the requirements"),
                        ("src/train.py", "train the model"),
                        ("src/evaluate.py", "evaluate the model")]:
        if need not in joined:
            print(f"WARNING: no command to {label}.")
            ok = False
    print()
    print("Nice work. Part 2 is done." if ok else "Fix the warnings or manually verify skipped commands, then run CHECK 4 again.")


# ----------------------------------------------------------------------
# CHECK 5: README numbers
# ----------------------------------------------------------------------
def check_numbers():
    _header("CHECK 5: Are the README's numbers true?")
    print("Training and scoring the model the way the code does today...")
    truth = _true_results()
    print()
    results = _section_text(README.read_text(), README_SECTIONS["Results"])
    if not results.strip():
        print("WARNING: no Results section to check. Finish Part 2 first.")
        return

    found = _extract_numbers(results)
    ok = True
    for metric in METRICS:
        value = _first(found, "overall", metric)
        if value is None:
            print(f"WARNING: Results doesn't report {metric}.")
            ok = False
        elif abs(value - truth["overall"][metric]) > TOLERANCE:
            print(f"WARNING: the {metric} in your README doesn't match what the project produces.")
            ok = False
        else:
            print(f"PASS: {metric} matches.")

    expected_test_count = truth["n_test_bookings"]
    results_text = _section_text(README.read_text(), README_SECTIONS["Results"])
    count_match = re.search(
        r"([\d,]+)\s+(?:held[- ]out\s+)?(?:test\s+)?bookings",
        results_text,
        re.I,
    )
    if count_match is None:
        print("WARNING: Results doesn't state how many held-out bookings were evaluated.")
        ok = False
    else:
        reported_count = int(count_match.group(1).replace(",", ""))
        if reported_count != expected_test_count:
            print(f"WARNING: Results reports {reported_count:,} held-out bookings; current code evaluates {expected_test_count:,}.")
            ok = False
        else:
            print(f"PASS: held-out test-set count ({expected_test_count:,}) matches.")

    stale = False
    if OLD_RESULTS.exists():
        try:
            saved = json.loads(OLD_RESULTS.read_text())
            stale = any(abs(float(saved.get(m, truth[m])) - truth[m]) > TOLERANCE for m in METRICS)
        except (ValueError, TypeError):
            stale = True
    if stale:
        print()
        print("WARNING: another file in this repo still reports numbers the code doesn't produce.")
        print("Anyone who reads it will be misled. Decide what to do with it.")
        ok = False

    print()
    if ok:
        print("PASS: the checked Results metrics and test-set count match.")
        print("This check does not validate every number or claim in the README.")
        print("Write down what you found and how you found it in WORKSHOP.md.")
    else:
        print("Where did those numbers come from? Can you prove which ones are true?")
        print("Copilot can help you look, but you decide what's right.")


# ----------------------------------------------------------------------
# Challenge: where does it struggle?
# ----------------------------------------------------------------------
def check_challenge():
    _header("CHALLENGE CHECK: Is your README honest about where the model struggles?")
    text = README.read_text()
    print("Training and scoring the model the way the code does today...")
    truth = _true_results()
    print()
    ok = True

    def report(status, label):
        nonlocal ok
        ok = ok and status == "PASS"
        print(f"  {status}: {label}")

    print("Results: overall numbers")
    found = _extract_numbers(_section_text(text, README_SECTIONS["Results"]))
    for metric in METRICS:
        value = _first(found, "overall", metric)
        right = value is not None and abs(value - truth["overall"][metric]) <= TOLERANCE
        report("PASS" if right else ("WRONG" if value is not None else "MISSING"), metric)

    limits = _section_text(text, README_SECTIONS["Limitations"])
    print("Limitations: recall by hotel")
    found = _extract_numbers(limits)
    for hotel in ["City Hotel", "Resort Hotel"]:
        value = _first(found, hotel, "recall")
        right = value is not None and abs(value - truth[hotel]["recall"]) <= TOLERANCE
        report("PASS" if right else ("WRONG" if value is not None else "MISSING"), f"{hotel} recall")

    print("Data source and license")
    credited = bool(re.search(r"antonio|data in brief", text, re.I)) and bool(re.search(r"cc[ -]?by", text, re.I))
    report("PASS" if credited else "MISSING", "credit and license")

    print("Claims a reviewer would push back on")
    claims = sorted({c for c in OVERCLAIMS if re.search(re.escape(c), text, re.I)})
    if claims:
        for c in claims:
            report("FOUND", f"\"{c}\". Cut it, or show the evidence.")
    elif not limits.strip():
        report("MISSING", "a Limitations section")
    else:
        report("PASS", "none found")

    print()
    if ok:
        print("Every item passes. Nice work.")
        print("Next, try More to Try in WORKSHOP.md, or start on your own repo.")
    else:
        print("Fix anything marked WRONG, MISSING, or FOUND, then run the check again.")


CHECKS = {
    "0": check_setup,
    "1": check_docstrings,
    "2": check_readme_sections,
    "3": check_requirements,
    "4": check_commands,
    "5": check_numbers,
    "challenge": check_challenge,
}

if __name__ == "__main__":
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "0"
    if which not in CHECKS:
        print(f"Unknown check '{which}'. Use one of: {', '.join(CHECKS)}")
        sys.exit(1)
    CHECKS[which]()
