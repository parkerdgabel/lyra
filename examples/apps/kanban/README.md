Kanban (Projects/Boards/Cards) — Lyra Example App

Overview
- SQLite-backed project boards with columns and cards.
- CLI to initialize DB, add projects/columns/cards, and list boards.
- Uses ArgsParse, DB Exec/SQL, and CLI formatting helpers (AnsiStyle, Rule, TableSimple).

Schema
- projects(id INTEGER PK, name TEXT)
- columns(id INTEGER PK, project_id INTEGER, name TEXT, pos INTEGER)
- cards(id INTEGER PK, project_id INTEGER, column_id INTEGER, title TEXT, desc TEXT, prio INTEGER, created_at INTEGER)

Commands
- init
- project add <name>
- column add <project_id> <name> [--pos N]
- card add <project_id> <column_id> <title> [--prio N]
- board <project_id>

Run
- lyra-runner --file examples/apps/kanban/cli.lyra -- init
- lyra-runner --file examples/apps/kanban/cli.lyra -- project add "Demo"
- lyra-runner --file examples/apps/kanban/cli.lyra -- column add 1 "Todo" --pos 1
- lyra-runner --file examples/apps/kanban/cli.lyra -- column add 1 "Doing" --pos 2
- lyra-runner --file examples/apps/kanban/cli.lyra -- column add 1 "Done" --pos 3
- lyra-runner --file examples/apps/kanban/cli.lyra -- card add 1 1 "Set up project" --prio 2
- lyra-runner --file examples/apps/kanban/cli.lyra -- board 1

