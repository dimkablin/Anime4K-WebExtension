import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const overlayManager = readFileSync('src/core/overlay-manager.ts', 'utf8');
const videoManager = readFileSync('src/core/video-manager.ts', 'utf8');
const videoEnhancer = readFileSync('src/core/video-enhancer.ts', 'utf8');

test('creating a player does not remove overlays owned by other players', () => {
  assert.doesNotMatch(overlayManager, /Detected orphaned overlay host/);
});

test('a playing video starts enhancement idempotently', () => {
  assert.match(videoManager, /event\.type === 'playing'/);
  assert.match(videoManager, /enhancer\?\.enableEnhancement\(\)/);
  assert.match(videoEnhancer, /public async enableEnhancement\(\)/);
  assert.match(videoEnhancer, /if \(this\.renderer \|\| this\.button\.disabled\) return;/);
});
