"use client";

import { useCallback, useEffect, useState } from "react";

/**
 * State mirrored to localStorage. Starts at `initial` on the server and the
 * first client render (so hydration matches), then loads the stored value.
 * Storage can throw (private mode, blocked site data); it is never required.
 */
export function useStored<T>(key: string, initial: T): [T, (v: T) => void] {
  const [value, setValue] = useState<T>(initial);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(key);
      // Reading browser-only storage after hydration is exactly what an effect is for.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      if (raw != null) setValue(JSON.parse(raw));
    } catch {}
  }, [key]);

  const set = useCallback(
    (v: T) => {
      setValue(v);
      try {
        localStorage.setItem(key, JSON.stringify(v));
      } catch {}
    },
    [key],
  );

  return [value, set];
}
