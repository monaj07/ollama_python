# Week 11 — Telegram image-upload guard

The `pi-telegram-plus` package exposes the general-purpose `tg_attach` tool. For this environment agent, `src/week_11/.pi/extensions/telegram_attachment_guard.ts` blocks model-invoked `tg_attach` calls unless **every** requested file is an existing observation photograph in `outputs/observations/`.

The guard runs at Pi's `tool_call` hook (before `tg_attach` executes) and validates the canonical filesystem paths with `realpath`, including symlink resolution. It permits up to 10 images in one call, including `scene.jpg` and `annotated.jpg` (also `.png`). An invalid or missing path blocks the *entire* call. It does not expose another tool to the LLM or alter memory/vision outputs. A failed image upload will produce a blocked tool response, not an arbitrary-file send.

This guard is intentionally scoped to **model-invoked `tg_attach`**. It does not restrict Pi's other tools, Telegram's UI commands, automatic outputs from the Telegram plugin, direct file operations by other extensions, or operating-system access. Keep the Pi tool allowlist narrow and Telegram bot restricted to your own paired account. Do not treat this as an OS sandbox.

## Test

From the repository root:

```bash
node --test src/week_11/tests/test_telegram_attachment_policy.mjs
```

Restart the Pi TUI afterward to load `.pi/extensions/telegram_attachment_guard.ts`. Test that a saved observation picture can be sent and that a request to attach an unrelated file is rejected. Both checks should be made via `tg_attach`, not through other Telegram commands.

## Speaker and Telegram traffic

Observed behaviour: completed local TUI answers speak over the selected Bluetooth sink, while Telegram turns produce text/images without local speech. This is **observed behaviour, not an enforced origin policy**. `pi-telegram-plus` may inject Telegram text as input with `source: "interactive"`, so an `event.source` comparison alone cannot reliably separate local from remote requests. This patch deliberately does not change the working audio extension. To guarantee separation independently of plugin internals, run local and Telegram agents in separately configured Pi processes with separate extension sets, or establish a supported explicit origin signal in the Telegram integration.
