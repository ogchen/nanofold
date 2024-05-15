import os
import glob
import logging
from functools import partial
from itertools import batched
from pathlib import Path
from tempfile import NamedTemporaryFile

from nanofold.common.residue_definitions import UNKNOWN_RESIDUE
from nanofold.common.residue_definitions import MSA_GAP
from nanofold.preprocess.msa_builder import execute_msa_search
from nanofold.preprocess.sto_parser import convert_to_a2m
from nanofold.preprocess.hhr_parser import parse_hhr


def get_chains_to_process(db_manager, msa_output_dir):
    chains = db_manager.chains().find({"templates": {"$exists": 0}}, {"_id": 1, "sequence": 1})
    search_glob = os.path.join(msa_output_dir, "*.pkl.gz")
    msa_files = glob.glob(search_glob)
    found_ids = [Path(m).stem.split(".")[0] for m in msa_files]
    for c in chains:
        if f"{c['_id']['structure_id']}_{c['_id']['chain_id']}" in found_ids:
            yield c


def preprocess_templates(templates):
    result = []
    for id, template in templates.items():
        if template["t_seq"] in template["q_seq"] or len(template["t_seq"]) < 10:
            continue
        positions = []
        adj_t_seq = ""
        counter = 0
        for t, q in zip(template["t_seq"], template["q_seq"]):
            if q != "-":
                if t != "-":
                    positions.append(template["target_start_pos"] + counter)
                    counter += 1
                else:
                    positions.append(None)
                adj_t_seq += t
            else:
                counter += 1
        result.append(
            (
                template["sum_prob"],
                {
                    "id": id.split("_"),
                    "seq": adj_t_seq,
                    "t_positions": positions,
                    "q_start": template["query_start_pos"],
                    "q_end": template["query_end_pos"],
                },
            )
        )
    result = sorted(result, key=lambda x: x[0], reverse=True)
    return [r[1] for r in result]


def count_differences(matched_positions, chain, template):
    differences = 0
    for i, pos in enumerate(matched_positions):
        if (
            pos is not None
            and chain["sequence"][pos] is not UNKNOWN_RESIDUE[0]
            and template["seq"][i] != chain["sequence"][pos]
        ):
            differences += 1
    return differences


def get_matched_positions(template, chain):
    def match_positions(positions):
        c_ind = 0
        matched_positions = []
        for t_pos in template["t_positions"]:
            if t_pos is not None:
                while c_ind < len(positions):
                    if positions[c_ind] < t_pos:
                        c_ind += 1
                    else:
                        break
                if c_ind < len(positions) and positions[c_ind] == t_pos:
                    matched_positions.append(c_ind)
                    continue
            matched_positions.append(None)
        return matched_positions

    matched_positions = match_positions(chain["label_positions"])
    differences = count_differences(matched_positions, chain, template)
    if differences / len(template["seq"]) > 0.05:
        raise ValueError(f"Unable to match positions for template {template['id']}")
    return matched_positions


def extract_template_features(templates, db_manager, query_length, max_templates):
    templates = preprocess_templates(templates)
    results = []
    for template in templates:
        chain = db_manager.chains().find_one(
            {"_id": {"structure_id": template["id"][0].lower(), "chain_id": template["id"][1]}},
            {"label_positions": 1, "sequence": 1, "translations": 1, "rotations": 1},
        )
        if chain is not None:
            pad_iterable = lambda x, fill: (
                fill * (template["q_start"] - 1) + x + fill * (query_length - template["q_end"])
            )
            positions = pad_iterable(get_matched_positions(template, chain), [None])
            sequence = pad_iterable(template["seq"], MSA_GAP)
            mask = [p is not None for p in positions]
            translations = [
                chain["translations"][i] if i is not None else [0.0] * 3 for i in positions
            ]
            rotations = [
                chain["rotations"][i] if i is not None else [[0.0] * 3] * 3 for i in positions
            ]
            results.append(
                {
                    "mask": mask,
                    "sequence": sequence,
                    "translations": translations,
                    "rotations": rotations,
                }
            )
            if len(results) >= max_templates:
                break
    return {
        "sequence": [r["sequence"] for r in results],
        "mask": [r["mask"] for r in results],
        "translations": [r["translations"] for r in results],
        "rotations": [r["rotations"] for r in results],
    }


def get_hhr(hhblits_runner, reformat_bin, msa_runner, chain):
    msa_sto = execute_msa_search(msa_runner, chain)
    with NamedTemporaryFile(mode="w") as a2m_file:
        convert_to_a2m(reformat_bin, msa_sto(), a2m_file.name)
        id = f"{chain['_id']['structure_id'].lower()}_{chain['_id']['chain_id']}"
        return hhblits_runner.run(a2m_file.name, id)


def get_hhr_contents(hhblits_runner, reformat_bin, msa_runner, executor, chains, batch_size=50):
    get_result = partial(get_hhr, hhblits_runner, reformat_bin, msa_runner)
    for i, batch in enumerate(batched(chains, batch_size)):
        result = executor.map(get_result, batch)
        for chain in batch:
            try:
                yield chain, next(result)
            except Exception as e:
                logging.error(
                    f"Failure while searching for templates for chain {chain['_id']}: {e}"
                )
                continue
        logging.info(f"Constructed template features for {i * batch_size + len(batch)} chains")


def build_template(
    hhblits_runner,
    reformat_bin,
    msa_runner,
    db_manager,
    executor,
    msa_output_dir,
    max_templates=20,
):
    chains = get_chains_to_process(db_manager, msa_output_dir)
    logging.info("Building template features")
    for chain, hhr_output in get_hhr_contents(
        hhblits_runner, reformat_bin, msa_runner, executor, chains
    ):
        try:
            templates = parse_hhr(hhr_output())
            features = extract_template_features(
                templates, db_manager, len(chain["sequence"]), max_templates
            )
            db_manager.chains().update_one(
                {"_id": chain["_id"]},
                {"$set": {"templates": features}},
            )
            logging.info("Built template features for chain %s", chain["_id"])
        except Exception as e:
            logging.error(f"Failed to build template features for chain {chain['_id']}: {e}")
