import logging
from functools import partial
from itertools import batched
from tempfile import NamedTemporaryFile


def get_chains_to_process(db_manager):
    chains = db_manager.chains().find({}, {"_id": 1, "sequence": 1})
    return list(chains)


def get_msa(msa_runner, chain):
    with NamedTemporaryFile(mode="w") as tmp:
        id = f"{chain['_id']['structure_id'].lower()}_{chain['_id']['chain_id'].upper()}"
        fasta = f">{id}\n{chain['sequence']}"
        tmp.writelines(fasta)
        tmp.flush()
        msa_runner.run(tmp.name, id)


def get_sto_contents(msa_runner, executor, chains, log_every=100):
    get_result = partial(get_msa, msa_runner)
    for i, batch in enumerate(batched(chains, log_every)):
        result = list(executor.map(get_result, batch))
        logging.info(
            f"Fetched small BFD alignments for {i * log_every + len(batch)}/{len(chains)} chains"
        )
        for chain, r in zip(batch, result):
            yield chain, r


def build_msa(msa_runner, db_manager, executor):
    chains = get_chains_to_process(db_manager)
    total_num_chains = db_manager.chains().count_documents({})
    logging.info(f"Found {len(chains)}/{total_num_chains} chains missing MSA")
    for chain, sto_contents in get_sto_contents(msa_runner, executor, chains):
        pass
