import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const read = (path) => readFileSync(path, 'utf8');

test('AnimeSR mode keeps native x4 output for 720p and 1080p', () => {
  const settings = read('src/utils/settings.ts');
  const renderer = read('src/core/animesr-renderer.ts');
  const host = read('native/animesr_host.py');

  assert.match(settings, /builtin-animesr-v2-tensorrt/);
  assert.match(renderer, /NATIVE_SCALE = 4/);
  assert.match(host, /SCALE = 4/);
  assert.match(renderer, /1920[^\n]+1080/);
  assert.match(host, /1920[^\n]+1080/);
  assert.doesNotMatch(host, /interpolate\s*\(/);
});

test('extension launches the backend through Native Messaging', () => {
  const manifest = JSON.parse(read('manifest.json'));
  const background = read('src/background.ts');

  assert.ok(manifest.permissions.includes('nativeMessaging'));
  assert.match(background, /chrome\.runtime\.connectNative\(ANIMESR_HOST_NAME\)/);
  assert.match(background, /ANIMESR_NATIVE_CONNECT/);
});

test('media server is local, token authenticated, and independent from TAS code', () => {
  const host = read('native/animesr_host.py');

  assert.match(host, /127\.0\.0\.1/);
  assert.match(host, /secrets\.token_urlsafe/);
  assert.doesNotMatch(host, /TheAnimeScripter|src\.upscale|src\.model/);
  assert.doesNotMatch(read('package.json'), /socket\.io/);
});

test('Windows installer registers the native host for Edge and Chrome', () => {
  const installer = read('native/install.ps1');

  assert.match(installer, /Microsoft\\Edge\\NativeMessagingHosts/);
  assert.match(installer, /Google\\Chrome\\NativeMessagingHosts/);
  assert.match(installer, /com\.dimkablin\.animesr/);
  assert.match(installer, /1080x1920\.engine/);
});
