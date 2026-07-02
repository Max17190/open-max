export function DiffView({ diff }: { diff: string }) {
  const lines = diff.split("\n").filter((l) => !l.startsWith("---") && !l.startsWith("+++"));
  return (
    <pre className="diff-body">
      {lines.map((line, i) => {
        const cls = line.startsWith("+")
          ? "dl-add"
          : line.startsWith("-")
            ? "dl-del"
            : line.startsWith("@@")
              ? "dl-hunk"
              : "dl-ctx";
        return (
          <div key={i} className={`dl ${cls}`}>
            {line || " "}
          </div>
        );
      })}
    </pre>
  );
}
