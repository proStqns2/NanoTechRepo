export class PoseBridge extends EventTarget {
  constructor({ video, captureWidth = 512, targetFps = 20, quality = 0.58 } = {}) {
    super();
    if (!(video instanceof HTMLVideoElement)) {
      throw new TypeError("PoseBridge requiert un élément vidéo.");
    }
    this.video = video;
    this.captureWidth = captureWidth;
    this.targetFps = Math.min(Math.max(targetFps, 1), 30);
    this.quality = Math.min(Math.max(quality, 0.2), 0.9);
    this.canvas = document.createElement("canvas");
    this.context = this.canvas.getContext("2d", { alpha: false, desynchronized: true });
    if (!this.context) {
      throw new Error("Impossible d'initialiser le canvas de capture.");
    }
    this.stream = null;
    this.socket = null;
    this.timer = 0;
    this.reconnectTimer = 0;
    this.reconnectDelay = 500;
    this.captureInProgress = false;
    this.awaitingPose = false;
    this.running = false;
    this.session = 0;
    this.lastSentAt = 0;
    this.receivedFrames = 0;
    this.fpsStartedAt = performance.now();
    this.endpoint = this._getEndpoint();
  }

  _getEndpoint() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${location.host}/ws/pose`;
  }

  async start() {
    if (this.running) return;
    this.running = true;
    const session = ++this.session;
    this._emit("status", { state: "camera", message: "Demande d’accès à la webcam…" });
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30, max: 30 },
        },
      });
      if (!this.running || session !== this.session) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      this.stream = stream;
      this.video.srcObject = stream;
      await this.video.play();
      if (!this.running || session !== this.session) return;
      this._emit("status", { state: "connecting", message: "Webcam active · connexion vision…" });
      this._connect(session);
    } catch (error) {
      if (session !== this.session) return;
      this.running = false;
      this._emit("error", {
        message: error?.name === "NotAllowedError"
          ? "Accès à la caméra refusé. Autorisez la webcam dans votre navigateur."
          : `Impossible d'initialiser la webcam : ${error instanceof Error ? error.message : "erreur inconnue"}`,
      });
      this._emit("status", { state: "error", message: "Erreur caméra" });
    }
  }

  stop() {
    this.running = false;
    this.session += 1;
    window.clearInterval(this.timer);
    window.clearTimeout(this.reconnectTimer);
    this.timer = 0;
    this.reconnectTimer = 0;
    this.captureInProgress = false;
    this.awaitingPose = false;
    if (this.socket) {
      this.socket.onclose = null;
      this.socket.onerror = null;
      this.socket.onmessage = null;
      this.socket.onopen = null;
      if (this.socket.readyState < WebSocket.CLOSING) this.socket.close(1000, "Session arrêtée");
      this.socket = null;
    }
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.video.srcObject = null;
    this._emit("status", { state: "offline", message: "SYSTÈME EN ATTENTE" });
  }

  _connect(session) {
    if (!this.running || session !== this.session) return;
    let socket;
    try {
      socket = new WebSocket(this.endpoint);
    } catch (error) {
      this._scheduleReconnect(session, error);
      return;
    }
    this.socket = socket;
    socket.binaryType = "arraybuffer";
    socket.onopen = () => {
      if (session !== this.session || socket !== this.socket) {
        socket.close();
        return;
      }
      this.reconnectDelay = 500;
      this._emit("status", { state: "online", message: "SERVEUR DE VISION EN LIGNE" });
      this._startCapture(session, socket);
    };
    socket.onmessage = (event) => {
      if (session !== this.session || socket !== this.socket) return;
      this._receive(event.data);
    };
    socket.onerror = () => {
      if (session === this.session) {
        this._emit("status", { state: "reconnecting", message: "Liaison vision interrompue…" });
      }
    };
    socket.onclose = (event) => {
      if (session !== this.session || socket !== this.socket) return;
      this.socket = null;
      this.awaitingPose = false;
      window.clearInterval(this.timer);
      this.timer = 0;
      if (this.running && event.code !== 1000) {
        this._scheduleReconnect(session, new Error(`WebSocket fermé (${event.code}).`));
      }
    };
  }

  _scheduleReconnect(session, error) {
    if (!this.running || session !== this.session || this.reconnectTimer) return;
    this._emit("status", { state: "reconnecting", message: "Reconnexion du serveur…" });
    this._emit("connection-error", { error });
    const delay = this.reconnectDelay;
    this.reconnectDelay = Math.min(this.reconnectDelay * 1.7, 8000);
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = 0;
      this._connect(session);
    }, delay);
  }

  _startCapture(session, socket) {
    window.clearInterval(this.timer);
    this.timer = window.setInterval(() => {
      void this._captureAndSend(session, socket);
    }, 1000 / this.targetFps);
  }

  async _captureAndSend(session, socket) {
    if (
      !this.running ||
      session !== this.session ||
      socket !== this.socket ||
      socket.readyState !== WebSocket.OPEN ||
      this.captureInProgress ||
      this.awaitingPose ||
      this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      socket.bufferedAmount > 900_000
    ) return;
    this.captureInProgress = true;
    try {
      const size = this._cropSize();
      this.canvas.width = size.width;
      this.canvas.height = size.height;
      const source = this._sourceCrop(size.aspect);
      this.context.drawImage(
        this.video,
        source.x,
        source.y,
        source.width,
        source.height,
        0,
        0,
        size.width,
        size.height,
      );
      const blob = await new Promise((resolve, reject) => {
        this.canvas.toBlob(
          (image) => image ? resolve(image) : reject(new Error("Encodage JPEG impossible.")),
          "image/jpeg",
          this.quality,
        );
      });
      if (
        this.running &&
        session === this.session &&
        socket === this.socket &&
        socket.readyState === WebSocket.OPEN &&
        socket.bufferedAmount < 900_000
      ) {
        socket.send(blob);
        this.lastSentAt = performance.now();
        this.awaitingPose = true;
        this._emit("frame-sent", { bytes: blob.size, at: this.lastSentAt });
      }
    } catch (error) {
      this._emit("error", { message: `Capture webcam impossible : ${error instanceof Error ? error.message : "erreur inconnue"}` });
    } finally {
      this.captureInProgress = false;
    }
  }

  _cropSize() {
    const rect = this.video.getBoundingClientRect();
    const aspect = rect.width > 0 && rect.height > 0 ? rect.width / rect.height : 4 / 3;
    const width = Math.min(this.captureWidth, Math.max(this.video.videoWidth, 1));
    const height = Math.max(1, Math.round(width / aspect));
    return { width, height, aspect };
  }

  _sourceCrop(targetAspect) {
    const sourceWidth = this.video.videoWidth;
    const sourceHeight = this.video.videoHeight;
    const sourceAspect = sourceWidth / Math.max(sourceHeight, 1);
    if (sourceAspect > targetAspect) {
      const width = sourceHeight * targetAspect;
      return { x: (sourceWidth - width) / 2, y: 0, width, height: sourceHeight };
    }
    const height = sourceWidth / targetAspect;
    return { x: 0, y: (sourceHeight - height) / 2, width: sourceWidth, height };
  }

  _receive(data) {
    let message;
    try {
      message = JSON.parse(typeof data === "string" ? data : new TextDecoder().decode(data));
    } catch (error) {
      this._emit("error", { message: `Réponse serveur illisible : ${error instanceof Error ? error.message : "JSON invalide"}` });
      return;
    }
    if (message.type === "error") {
      this._emit("error", { message: message.message || "Erreur du serveur de vision." });
      return;
    }
    if (message.type !== "pose") return;
    this.awaitingPose = false;
    this.receivedFrames += 1;
    const now = performance.now();
    const elapsed = now - this.fpsStartedAt;
    if (elapsed >= 1000) {
      this._emit("fps", { value: Math.round(this.receivedFrames * 1000 / elapsed) });
      this.receivedFrames = 0;
      this.fpsStartedAt = now;
    }
    this._emit("pose", { ...message, roundTripMs: this.lastSentAt ? now - this.lastSentAt : null });
  }

  _emit(type, detail) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }
}
