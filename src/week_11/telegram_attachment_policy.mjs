/**
 * Fail-closed policy for model-invoked Telegram uploads.
 * Only existing observation photographs may be attached with tg_attach.
 * This validates canonical paths, so symlinks cannot escape the evidence root.
 */
import { realpathSync, statSync } from 'node:fs';
import { basename, isAbsolute, join, relative, sep } from 'node:path';

const ALLOWED_IMAGES = new Set(['scene.jpg', 'annotated.jpg', 'scene.png', 'annotated.png']);

export function validateObservationAttachments(input, cwd) {
  if (!input || typeof input !== 'object' || Array.isArray(input) ||
      !Array.isArray(input.paths) || input.paths.length === 0 || input.paths.length > 10) {
    return { ok: false, reason: 'tg_attach expects 1–10 observation image paths' };
  }

  let evidenceRoot;
  try {
    evidenceRoot = realpathSync(join(cwd, '..', '..', 'outputs', 'observations'));
  } catch {
    return { ok: false, reason: 'Observation-image directory is unavailable' };
  }

  for (const requestedPath of input.paths) {
    if (typeof requestedPath !== 'string' || !isAbsolute(requestedPath)) {
      return { ok: false, reason: 'Attachments require absolute observation-image paths' };
    }
    try {
      const path = realpathSync(requestedPath);
      const subpath = relative(evidenceRoot, path);
      if (!subpath || subpath === '..' || subpath.startsWith(`..${sep}`) || isAbsolute(subpath) ||
          !ALLOWED_IMAGES.has(basename(path)) || !statSync(path).isFile()) {
        return { ok: false, reason: 'Only saved observation photographs may be attached' };
      }
    } catch {
      return { ok: false, reason: 'An observation image does not exist or cannot be accessed' };
    }
  }
  return { ok: true };
}
