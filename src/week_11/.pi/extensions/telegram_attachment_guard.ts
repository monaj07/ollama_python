/**
 * Guard the pi-telegram-plus tg_attach tool at the actual tool-call boundary.
 * The model cannot bypass this with a prompt or an invented file path.
 * This does not change Telegram inbound voice or local Bluetooth speech.
 */
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
// @ts-ignore -- The policy is a small native ESM module tested independently with node.
import { validateObservationAttachments } from '../../telegram_attachment_policy.mjs';

export default function (pi: ExtensionAPI) {
  pi.on('tool_call', (event, ctx) => {
    if (event.toolName !== 'tg_attach') return;
    const verdict = validateObservationAttachments(event.input, ctx.cwd);
    if (!verdict.ok) {
      return { block: true, reason: verdict.reason };
    }
  });
}
