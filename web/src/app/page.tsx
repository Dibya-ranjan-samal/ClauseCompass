import { Suspense } from "react";
import AskView from "./AskView";

export default function Page() {
  return (
    <Suspense>
      <AskView />
    </Suspense>
  );
}
