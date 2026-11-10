interface Props {
  rank: number;
  firstStageRank: number | null;
}

/** "▲ 4" when phase 2 moved a patent up from phase-1 rank #5 to #1. */
export function RankChange({ rank, firstStageRank }: Props) {
  if (firstStageRank === null) return null;
  const moved = firstStageRank - rank;
  const title = `Phase 1 rank #${firstStageRank}, after reranking #${rank}`;
  if (moved === 0)
    return (
      <span className="rank-change same" title={title}>
        = #{firstStageRank}
      </span>
    );
  return (
    <span className={`rank-change ${moved > 0 ? "up" : "down"}`} title={title}>
      {moved > 0 ? "▲" : "▼"} {Math.abs(moved)}
      <span className="rank-change-from">from #{firstStageRank}</span>
    </span>
  );
}
