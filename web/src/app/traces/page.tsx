import { Suspense } from "react";
import TracesView from "./TracesView";

export default function Page() {
  return (
    <Suspense>
      <TracesView />
    </Suspense>
  );
}
