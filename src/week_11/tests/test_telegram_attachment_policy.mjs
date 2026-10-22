import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, symlinkSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { validateObservationAttachments } from '../telegram_attachment_policy.mjs';

function setup() {
  const root = mkdtempSync(join(tmpdir(), 'pi-telegram-guard-'));
  const cwd = join(root, 'src', 'week_11');
  const evidence = join(root, 'outputs', 'observations', 'capture-id');
  mkdirSync(cwd, { recursive: true });
  mkdirSync(evidence, { recursive: true });
  const image = join(evidence, 'scene.jpg');
  writeFileSync(image, 'fake-image');
  return { root, cwd, evidence, image, cleanup: () => rmSync(root, { recursive: true, force: true }) };
}

test('allows canonical stored observation image', () => {
  const ctx = setup();
  try { assert.deepEqual(validateObservationAttachments({paths:[ctx.image]}, ctx.cwd), {ok:true}); }
  finally { ctx.cleanup(); }
});

test('allows a batch of stored observation images', () => {
  const ctx = setup();
  try {
    const second = join(ctx.evidence, 'annotated.jpg');
    writeFileSync(second, 'fake-image');
    assert.equal(validateObservationAttachments({paths:[ctx.image,second]}, ctx.cwd).ok,true);
  } finally { ctx.cleanup(); }
});

test('blocks unrelated files even if they exist', () => {
  const ctx = setup();
  try {
    const other = join(ctx.root, 'private.txt');
    writeFileSync(other, 'private');
    assert.equal(validateObservationAttachments({paths:[other]}, ctx.cwd).ok,false);
  } finally { ctx.cleanup(); }
});

test('blocks image files outside the evidence directory', () => {
  const ctx = setup();
  try {
    const outside = join(ctx.root, 'scene.jpg');
    writeFileSync(outside, 'not evidence');
    assert.equal(validateObservationAttachments({paths:[outside]}, ctx.cwd).ok,false);
  } finally { ctx.cleanup(); }
});

test('blocks symlink escape from evidence directory', () => {
  const ctx = setup();
  try {
    const outside = join(ctx.root, 'scene.jpg');
    writeFileSync(outside, 'not evidence');
    const escape = join(ctx.evidence, 'annotated.jpg');
    symlinkSync(outside, escape);
    assert.equal(validateObservationAttachments({paths:[escape]}, ctx.cwd).ok,false);
  } finally { ctx.cleanup(); }
});

test('blocks voice and other non-photographic attachments', () => {
  const ctx = setup();
  try {
    const voice = join(ctx.evidence, 'message.oga');
    writeFileSync(voice, 'voice');
    assert.equal(validateObservationAttachments({paths:[voice]}, ctx.cwd).ok,false);
  } finally { ctx.cleanup(); }
});

test('rejects missing images, absent paths, relative paths, excessive batch size', () => {
  const ctx = setup();
  try {
    for (const input of [
      {paths:[join(ctx.evidence, 'missing.jpg')]},
      {paths:[]},
      {paths:['capture-id/scene.jpg']},
      {paths:new Array(11).fill(ctx.image)},
      {path:ctx.image},
      null,
    ]) assert.equal(validateObservationAttachments(input, ctx.cwd).ok,false);
  } finally { ctx.cleanup(); }
});

test('blocks entire batch if any image path is invalid', () => {
  const ctx = setup();
  try { assert.equal(validateObservationAttachments({paths:[ctx.image, '/etc/passwd']}, ctx.cwd).ok,false); }
  finally { ctx.cleanup(); }
});
