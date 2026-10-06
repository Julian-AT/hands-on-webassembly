'use client';
import { useEffect, useRef, useState } from 'react';
import { AssignmentLauncher } from '../../runtime/launcher-session.mjs';
import manifest from '../lib/assignment-sessions.json';
import { selection, selectionUrl } from '../lib/navigation.mjs';
const titles = [
  'Tabular data',
  'Data types',
  'Unsupervised learning',
  'Supervised learning',
  'Logistic regression',
  'Neural networks',
  'Convolutional neural networks',
];
export default function Course() {
  const container = useRef(null),
    launcher = useRef(null);
  const [unit, setUnit] = useState(1),
    [status, setStatus] = useState({ state: 'loading', unit: 1 });
  useEffect(() => {
    let live = true;
    const instance = new AssignmentLauncher({
      window,
      container: container.current,
      manifest,
      onStatus: (value) => {
        if (live && value.state !== 'disposed') setStatus(value);
      },
    });
    launcher.current = instance;
    const restore = () => {
      const current = selection(window.location.href);
      if (current.normalize) window.history.replaceState(null, '', current.url);
      setUnit(current.unit);
      instance
        .select(current.unit)
        .catch(
          (error) =>
            live && setStatus({ state: 'error', error: error.message }),
        );
    };
    const leave = () => {
      instance.close();
    };
    const resume = (event) => {
      if (event.persisted) window.location.reload();
    };
    restore();
    window.addEventListener('popstate', restore);
    window.addEventListener('pagehide', leave);
    window.addEventListener('pageshow', resume);
    return () => {
      live = false;
      window.removeEventListener('popstate', restore);
      window.removeEventListener('pagehide', leave);
      window.removeEventListener('pageshow', resume);
      instance.close();
      launcher.current = null;
    };
  }, []);
  const choose = (event) => {
    const next = Number(event.target.value);
    if (next === unit) return;
    window.history.pushState(
      null,
      '',
      selectionUrl(window.location.href, next),
    );
    setUnit(next);
    setStatus({ state: 'loading', unit: next });
    launcher.current
      ?.select(next)
      .catch((error) => setStatus({ state: 'error', error: error.message }));
  };
  return (
    <main className="course">
      <header>
        <label htmlFor="assignment">Assignment</label>
        <select id="assignment" value={unit} onChange={choose}>
          {titles.map((title, i) => (
            <option key={i} value={i + 1}>
              {i + 1} — {title}
            </option>
          ))}
        </select>
        <p className="status" role="status" aria-live="polite">
          {status.state === 'error'
            ? status.error
            : status.state === 'ready'
              ? `Assignment ${unit} ready`
              : `Loading assignment ${unit}…`}
        </p>
        {status.state === 'error' && (
          <button onClick={() => launcher.current?.retry()}>Retry</button>
        )}
      </header>
      <div
        className="frame"
        ref={container}
        aria-label="Assignment workspace"
      />
    </main>
  );
}
