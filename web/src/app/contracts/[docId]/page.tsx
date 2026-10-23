import { Suspense } from "react";
import ContractView from "./ContractView";

export default function Page() {
  return (
    <Suspense>
      <ContractView />
    </Suspense>
  );
}
