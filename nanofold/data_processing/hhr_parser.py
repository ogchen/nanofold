def parse_hhr(hhr_contents):
    lines = hhr_contents.splitlines()
    result = {}

    current_id, query = None, None
    for i, line in enumerate(lines):
        if line.startswith("Query"):
            query = line.split()[1]
        if line.startswith(">"):
            current_id = line.split()[0][1:]
            sum_prob = float([x for x in lines[i + 1].split() if "Sum_probs" in x][0].split("=")[1])
            result[current_id] = {
                "sum_prob": sum_prob,
                "t_seq": "",
                "q_seq": "",
            }
        if current_id is not None and line.startswith(f"T {current_id}"):
            l = line.split()
            r = result[current_id]

            r["t_seq"] += l[3]
            r["target_start_pos"] = (
                int(l[2]) if "target_start_pos" not in r else r["target_start_pos"]
            )
            r["target_end_pos"] = int(l[4])
        if current_id is not None and line.startswith(f"Q {query}"):
            l = line.split()
            r = result[current_id]

            r["q_seq"] += l[3]
            r["query_start_pos"] = int(l[2]) if "query_start_pos" not in r else r["query_start_pos"]
            r["query_end_pos"] = int(l[4])
    return {k: v for k, v in result.items() if k.lower() != query.lower()}
