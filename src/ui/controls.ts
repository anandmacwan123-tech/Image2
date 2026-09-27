// Bars, labels and sliders. Every control is a single word; state shows as
// colour and a 1 px underline, never weight.

export function press(buttons: Iterable<Element>, active: (b: HTMLElement) => boolean): void {
  for (const b of buttons) b.setAttribute("aria-pressed", String(active(b as HTMLElement)));
}

export interface Slider {
  value: number;
  set(v: number): void;
}

/**
 * A word followed by a 1 px track. Drag anywhere on it to set the value;
 * double click resets.
 */
export function slider(el: HTMLElement, min: number, max: number, initial: number, onInput: (v: number) => void): Slider {
  const track = el.querySelector<HTMLElement>(".track")!;
  const fill = el.querySelector<HTMLElement>(".fill")!;
  const state: Slider = {
    value: initial,
    set(v: number) {
      state.value = Math.min(max, Math.max(min, v));
      fill.style.width = `${((state.value - min) / (max - min)) * 100}%`;
    },
  };
  state.set(initial);
  let dragging = false;
  const update = (ev: PointerEvent) => {
    const r = track.getBoundingClientRect();
    const t = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width));
    state.set(min + t * (max - min));
    onInput(state.value);
  };
  el.addEventListener("pointerdown", (ev) => {
    dragging = true;
    el.setPointerCapture(ev.pointerId);
    el.classList.add("active");
    update(ev);
  });
  el.addEventListener("pointermove", (ev) => dragging && update(ev));
  const end = () => {
    dragging = false;
    el.classList.remove("active");
  };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", end);
  el.addEventListener("dblclick", () => {
    state.set(initial);
    onInput(state.value);
  });
  return state;
}

/** The 1 px line across the top edge. */
export class ProgressLine {
  private hideTimer = 0;
  constructor(private el: HTMLElement) {}
  set(fraction: number): void {
    clearTimeout(this.hideTimer);
    this.el.style.opacity = "1";
    this.el.style.width = `${Math.min(100, Math.max(0, fraction * 100))}%`;
  }
  done(): void {
    this.set(1);
    this.hideTimer = window.setTimeout(() => {
      this.el.style.opacity = "0";
      window.setTimeout(() => (this.el.style.width = "0"), 400);
    }, 250);
  }
}
