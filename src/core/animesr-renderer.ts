export const NATIVE_SCALE = 4;

const INPUT_WIDTH = 1280;
const INPUT_HEIGHT = 720;
const MIME_TYPE = 'video/mp4; codecs="av01.0.13M.08"';

interface NativeConnectReply {
  ok: boolean;
  endpoint?: string;
  error?: string;
}

interface RendererOptions {
  video: HTMLVideoElement;
  output: HTMLVideoElement;
  onError: (error: Error) => void;
  onFirstFrameRendered: () => void;
}

interface ServerMessage {
  type: 'started' | 'ready' | 'error' | 'stats';
  error?: string;
  outputWidth?: number;
  outputHeight?: number;
}

export class AnimeSRRenderer {
  private readonly mediaSource = new MediaSource();
  private readonly captureCanvas = new OffscreenCanvas(INPUT_WIDTH, INPUT_HEIGHT);
  private readonly captureContext = this.captureCanvas.getContext('2d', { willReadFrequently: true })!;
  private readonly segments: Uint8Array[] = [];
  private readonly objectUrl: string;
  private socket: WebSocket | null = null;
  private sourceBuffer: SourceBuffer | null = null;
  private frameRequest: number | null = null;
  private frameInFlight = false;
  private connected = false;
  private destroyed = false;

  private constructor(private options: RendererOptions) {
    this.objectUrl = URL.createObjectURL(this.mediaSource);
    this.options.output.src = this.objectUrl;
  }

  static async create(options: RendererOptions): Promise<AnimeSRRenderer> {
    if (!MediaSource.isTypeSupported(MIME_TYPE)) {
      throw new Error('Edge cannot decode the AV1 stream required by AnimeSR.');
    }
    AnimeSRRenderer.assertInputSize(options.video);

    const renderer = new AnimeSRRenderer(options);
    await renderer.initialize();
    return renderer;
  }

  private static assertInputSize(video: HTMLVideoElement): void {
    if (video.videoWidth !== INPUT_WIDTH || video.videoHeight !== INPUT_HEIGHT) {
      throw new Error(`AnimeSR TensorRT currently requires ${INPUT_WIDTH}x${INPUT_HEIGHT} input.`);
    }
  }

  private async initialize(): Promise<void> {
    this.mediaSource.addEventListener('sourceopen', () => this.openSourceBuffer(), { once: true });
    const native = await chrome.runtime.sendMessage({ type: 'ANIMESR_NATIVE_CONNECT' }) as NativeConnectReply;
    if (!native.ok || !native.endpoint) {
      throw new Error(native.error || 'AnimeSR native host is not installed.');
    }

    await this.connectMediaSocket(native.endpoint);
    this.options.output.addEventListener('playing', this.options.onFirstFrameRendered, { once: true });
    this.options.video.addEventListener('seeking', this.resetTemporalState);
    this.requestNextFrame();
  }

  private connectMediaSocket(endpoint: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(endpoint);
      const timeout = window.setTimeout(() => {
        socket.close();
        reject(new Error('AnimeSR media connection timed out.'));
      }, 15000);
      socket.binaryType = 'arraybuffer';
      socket.onerror = () => reject(new Error('Could not connect to the AnimeSR media server.'));
      socket.onmessage = (event) => this.handleSocketMessage(event, resolve, reject, timeout);
      socket.onclose = () => {
        if (!this.destroyed) this.fail(new Error('AnimeSR media server disconnected.'));
      };
      socket.onopen = () => socket.send(JSON.stringify({
        type: 'start',
        width: INPUT_WIDTH,
        height: INPUT_HEIGHT,
        fps: 24,
      }));
      this.socket = socket;
    });
  }

  private handleSocketMessage(
    event: MessageEvent,
    resolve: () => void,
    reject: (error: Error) => void,
    timeout: number,
  ): void {
    if (event.data instanceof ArrayBuffer) {
      this.queueSegment(new Uint8Array(event.data));
      return;
    }

    const message = JSON.parse(String(event.data)) as ServerMessage;
    if (message.type === 'ready') {
      this.frameInFlight = false;
      return;
    }
    if (message.type === 'error') {
      clearTimeout(timeout);
      const error = new Error(message.error || 'AnimeSR native host failed.');
      if (this.connected) this.fail(error);
      else reject(error);
      return;
    }
    if (message.type !== 'started') return;

    clearTimeout(timeout);
    if (message.outputWidth !== INPUT_WIDTH * NATIVE_SCALE || message.outputHeight !== INPUT_HEIGHT * NATIVE_SCALE) {
      reject(new Error('AnimeSR native host returned an unexpected output size.'));
      return;
    }
    this.connected = true;
    resolve();
  }

  private openSourceBuffer(): void {
    if (this.destroyed) return;
    this.sourceBuffer = this.mediaSource.addSourceBuffer(MIME_TYPE);
    this.sourceBuffer.mode = 'sequence';
    this.sourceBuffer.addEventListener('updateend', () => {
      this.followLiveEdge();
      this.appendNextSegment();
    });
    this.appendNextSegment();
  }

  private queueSegment(segment: Uint8Array): void {
    if (this.destroyed) return;
    this.segments.push(segment);
    this.appendNextSegment();
  }

  private appendNextSegment(): void {
    if (!this.sourceBuffer || this.sourceBuffer.updating || this.segments.length === 0) return;
    try {
      this.sourceBuffer.appendBuffer(this.segments.shift()! as BufferSource);
      void this.options.output.play();
    } catch (error) {
      this.fail(error as Error);
    }
  }

  private followLiveEdge(): void {
    const buffered = this.options.output.buffered;
    if (buffered.length === 0) return;
    const liveEdge = buffered.end(buffered.length - 1);
    if (liveEdge - this.options.output.currentTime > 0.5) {
      this.options.output.currentTime = Math.max(0, liveEdge - 0.1);
    }
  }

  private requestNextFrame(): void {
    if (this.destroyed) return;
    this.frameRequest = this.options.video.requestVideoFrameCallback(() => {
      this.captureFrame();
      this.requestNextFrame();
    });
  }

  private captureFrame(): void {
    if (this.frameInFlight || this.options.video.paused || this.socket?.readyState !== WebSocket.OPEN) return;
    try {
      this.captureContext.drawImage(this.options.video, 0, 0, INPUT_WIDTH, INPUT_HEIGHT);
      const rgba = this.captureContext.getImageData(0, 0, INPUT_WIDTH, INPUT_HEIGHT).data.buffer;
      this.frameInFlight = true;
      this.socket.send(rgba);
    } catch (error) {
      this.fail(new Error(`The player blocks frame capture: ${(error as Error).message}`));
    }
  }

  private readonly resetTemporalState = (): void => {
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify({ type: 'reset' }));
    this.segments.length = 0;
    this.frameInFlight = false;
  };

  private fail(error: Error): void {
    if (!this.destroyed) this.options.onError(error);
  }

  async updateVideoSource(video: HTMLVideoElement): Promise<void> {
    AnimeSRRenderer.assertInputSize(video);
    this.options.video.removeEventListener('seeking', this.resetTemporalState);
    this.options.video = video;
    this.options.video.addEventListener('seeking', this.resetTemporalState);
    this.resetTemporalState();
  }

  async updateConfiguration(): Promise<void> {
    // AnimeSR v2 uses one fixed FP16 TensorRT profile and its native x4 output.
  }

  destroy(): void {
    this.destroyed = true;
    this.options.video.removeEventListener('seeking', this.resetTemporalState);
    if (this.frameRequest !== null) this.options.video.cancelVideoFrameCallback(this.frameRequest);
    this.socket?.close();
    this.options.output.pause();
    this.options.output.removeAttribute('src');
    this.options.output.load();
    URL.revokeObjectURL(this.objectUrl);
    this.segments.length = 0;
  }
}
