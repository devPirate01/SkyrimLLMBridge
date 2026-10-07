# 🎮 Skyrim Plugin — Installation Guide

## What's in This Folder

This directory contains the compiled Skyrim plugin files that form the **game-engine side** of the Skyrim LLM Bridge. These files are copied directly into your Skyrim installation — no mod manager compilation required.

> **Note:** These are compiled, ready-to-install files. The Papyrus source code (`.psc` files) was developed in the **Creation Kit** and can be exported from the `.esp` if you wish to inspect or modify the scripts.

---

## Files Included

| File | Install Path (relative to Skyrim `Data/`) | Purpose |
| :--- | :--- | :--- |
| `CompanionLLMV01.esp` | `Data/` | Main plugin — defines the Poisoned Mead quest, NPC aliases, and dialogue topics |
| `Seq/CompanionLLMV01.seq` | `Data/Seq/` | Start-Enabled Quest file — ensures the quest begins automatically on game load |
| `Scripts/MMJ_PoisonedMeadQuestScript.pex` | `Data/Scripts/` | Compiled quest controller — manages quest stage progression |
| `Scripts/MMJ_TavernWitnessChatLauncher.pex` | `Data/Scripts/` | Compiled chat launcher — reads `response.json` and triggers NPC dialogue |
| `Scripts/QF_MMJ_PoisonedMeadQuest_*.pex` | `Data/Scripts/` | Auto-generated quest fragment (Creation Kit output) |
| `Scripts/TIF__*.pex` | `Data/Scripts/` | Auto-generated topic info fragments (dialogue line scripts) |

---

## Prerequisites

Install these **before** copying the plugin files. All are available free on [Nexus Mods](https://www.nexusmods.com/skyrimspecialedition):

| # | Mod | Why it's needed |
| :--- | :--- | :--- |
| 1 | **[SKSE64](https://skse.silverlock.org/)** | Skyrim Script Extender — required to run all Papyrus extensions. Launch the game via `skse64_loader.exe`, never the standard launcher. |
| 2 | **[Address Library for SKSE Plugins](https://www.nexusmods.com/skyrimspecialedition/mods/32444)** — All In One | Provides version-independent address offsets required by virtually all SKSE-based mods below. |
| 3 | **[SkyUI](https://www.nexusmods.com/skyrimspecialedition/mods/12604)** | Powers the in-game UI and MCM (Mod Configuration Menu) system used by the dialogue interface. |
| 4 | **[PapyrusUtil AE/SE](https://www.nexusmods.com/skyrimspecialedition/mods/13048)** — Scripting Utility Functions | Provides `JsonUtil` and `StorageUtil` — the JSON file I/O layer the mod uses to pass game state to the Python bridge and receive responses. |
| 5 | **[Papyrus Extender](https://www.nexusmods.com/skyrimspecialedition/mods/22854)** | Extends the Papyrus scripting engine with additional functions used by the dialogue and quest scripts. |
| 6 | **[DbMiscFunctions](https://www.nexusmods.com/skyrimspecialedition/mods/18901)** | Utility scripting library required by Extended Vanilla Menus and related UI components. |
| 7 | **[Extended Vanilla Menus](https://www.nexusmods.com/skyrimspecialedition/mods/49900)** | Extends Skyrim's native dialogue and menu UI — required for the free-text player input interface. |
| 8 | **[ConsoleUtilSSE NG](https://www.nexusmods.com/skyrimspecialedition/mods/76649)** | Provides console utility functions used internally by the scripting stack at runtime. |

> **⚠️ Important:** The LLM Bridge Python backend (`bridge.py`) must be **already running** before you launch Skyrim. If it is not active, NPC dialogue will silently produce no response.

> **Why my initial forensic scan missed these:** The `.esp` master file list only captures hard-declared ESP dependencies. Mods like SkyUI, Papyrus Extender, and Address Library are *soft runtime dependencies* — they must be installed and active for the UI and extended Papyrus functions to work, but the engine does not enforce their presence at load time.


---

## Installation Steps

### Step 1 — Copy Plugin Files

Copy the contents of this folder into your Skyrim `Data` directory, preserving the subdirectory structure:

```
skyrim_plugin/
├── CompanionLLMV01.esp       →  <Skyrim>/Data/CompanionLLMV01.esp
├── Seq/
│   └── CompanionLLMV01.seq   →  <Skyrim>/Data/Seq/CompanionLLMV01.seq
└── Scripts/
    └── *.pex                 →  <Skyrim>/Data/Scripts/*.pex
```

Where `<Skyrim>` is your Skyrim installation directory, typically:
```
C:\Program Files (x86)\Steam\steamapps\common\Skyrim Special Edition\
```

Or wherever your Steam library is located.

### Step 2 — Configure the Python Bridge

In the repository root, configure your `.env` file to point `SKYRIM_DATA_DIR` at the StorageUtil data folder:

```ini
SKYRIM_DATA_DIR=C:\Program Files (x86)\Steam\steamapps\common\Skyrim Special Edition\Data\SKSE\Plugins\StorageUtilData\CompanionLLM
```

StorageUtil automatically creates this folder on first game launch if it doesn't already exist.

### Step 3 — Start the Bridge

From the repository root, run:

```bash
python bridge.py
```

Leave this terminal window open. You will see `Waiting for request.json...` when it is ready.

### Step 4 — Launch Skyrim via SKSE

Start the game using `skse64_loader.exe` (not the standard Skyrim launcher). Enable `CompanionLLMV01.esp` in your load order if it is not already active.

### Step 5 — In Game

Travel to **The Bannered Mare** in **Whiterun** and speak to **Runa** (the tavern witness). Initiate conversation — the mod will intercept your dialogue and route it through the LLM bridge in real time.

---

## ⚠️ Proof-of-Concept Notice

This plugin was built and tested on a **specific modded Skyrim installation**. It references a custom NPC setup (`tavern_witness` / Runa) in The Bannered Mare. As such:

- It is designed as a **research prototype and proof of concept**, not a polished consumer mod.
- It does **not** conflict with vanilla NPCs but may behave unexpectedly if you have other mods heavily altering The Bannered Mare.
- The plugin's Form IDs start with `02` (mod index), so load order position matters if you have many mods.
- **If you have trouble setting up the game environment** (registry paths, SKSE install, StorageUtil paths), please contact the author — setup is straightforward once the paths are correct, but Skyrim's directory structure varies between Steam library locations.

---

## Contact

If you run into setup issues, feel free to reach out:
- **Email:** Jahangirim@cardiff.ac.uk
- **GitHub Issues:** Open an issue on this repository
