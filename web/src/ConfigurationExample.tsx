import type { ReactNode } from "react";

export function ConfigurationExample({
  title,
  children,
  value,
}: {
  title: string;
  children: ReactNode;
  value: unknown;
}) {
  return (
    <details className="configuration-example" open>
      <summary>{title}</summary>
      <div className="example-description">{children}</div>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}
