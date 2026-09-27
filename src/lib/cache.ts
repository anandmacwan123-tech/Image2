// Geometry cache keyed by image hash and model version, so reopening an image
// skips inference. The light fit is cheap and always reruns, so fit changes
// apply to cached images too. IndexedDB works in the worker as well.

const DB = "depth-light";
const STORE = "geometry";

export interface CachedGeometry {
  width: number;
  height: number;
  depth: Float32Array;
  normals: Float32Array;
  mask: Uint8Array;
  fx: number;
  fy: number;
  metricScale: number;
}

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function hashBytes(bytes: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

export async function getCached(key: string): Promise<CachedGeometry | undefined> {
  try {
    const db = await open();
    return await new Promise((resolve, reject) => {
      const req = db.transaction(STORE, "readonly").objectStore(STORE).get(key);
      req.onsuccess = () => resolve(req.result as CachedGeometry | undefined);
      req.onerror = () => reject(req.error);
    });
  } catch {
    return undefined;
  }
}

export async function putCached(key: string, value: CachedGeometry): Promise<void> {
  try {
    const db = await open();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).put(value, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    // A full or blocked store only costs a rerun next time.
  }
}
