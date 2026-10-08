from itertools import islice
from datetime import timedelta
from typing import Any, Dict

from cognite.client import CogniteClient
from cognite.client.data_classes.data_modeling import NodeId, ViewId
from cognite.client.data_classes.data_modeling.cdm.v1 import CogniteAsset, CogniteTimeSeries, CogniteTimeSeriesApply
from cognite.client.data_classes.filters import Prefix, ContainsAny
from cognite.client.exceptions import CogniteNotFoundError

import numpy as np

from cognite.client.config import global_config
global_config.disable_pypi_version_check = True


def batcher(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def get_time_series_for_site(client: CogniteClient, site, space):
    this_site = site.lower()
    sub_tree_root = client.data_modeling.instances.retrieve_nodes(
        NodeId(space, this_site),
        node_cls=CogniteAsset
    )

    if not sub_tree_root:
        print(
            f"----No CogniteAssets in CDF for {site}!----\n"
            f"    Run the 'Create Cognite Asset Hierarchy' transformation!"
        )
        return []

    sub_tree_nodes = client.data_modeling.instances.list(
        instance_type=CogniteAsset,
        filter=Prefix(property=["cdf_cdm", "CogniteAsset/v1", "path"], value=sub_tree_root.path),
        limit=None
    )

    if not sub_tree_nodes:
        print(
            f"----No CogniteTimeSeries in CDF for {site}!----\n"
            f"    Run the 'Contextualize Timeseries and Assets' transformation!"
        )
        return []

    value_list = [{"space": node.space, "externalId": node.external_id} for node in sub_tree_nodes]

    time_series = [
        client.data_modeling.instances.search(
            view=ViewId("cdf_cdm", "CogniteTimeSeries", "v1"),
            instance_type=CogniteTimeSeries,
            space=space,
            filter=ContainsAny(property=["cdf_cdm", "CogniteTimeSeries/v1", "assets"], values=batch),
            limit=None
        )
        for batch in batcher(value_list, 20)
    ]

    # Combine list of batch results into a single NodeList
    time_series = [node for nodelist in time_series for node in nodelist]

    if not time_series:
        print("No CogniteTimeSeries in the CogniteCore Data Model (cdf_cdm Space)")
        return []

    return time_series

def handle(client: CogniteClient, data: Dict[str, Any] | None = None) -> None:
    lookback_minutes = None
    sites = None
    if data is None:
        data = {}

    if data:
        lookback_minutes = timedelta(minutes=data.get("lookback_minutes", 60)).total_seconds() * 1000
        sites = data.get("sites")

    all_sites = [
        "Houston",
        "Oslo",
        "Kuala_Lumpur",
        "Hannover",
        "Nuremberg",
        "Marseille",
        "Sao_Paulo",
        "Chicago",
        "Rotterdam",
        "London",
    ]

    lookback_minutes = lookback_minutes or timedelta(minutes=60).total_seconds() * 1000
    sites = sites or all_sites

    print(f"Processing datapoints for these sites: {sites}")
    # The OEE job creates and inserts derived time series in the same target space.
    # Running several site jobs concurrently causes race conditions where one thread
    # creates a derived time series after another thread has already tried to insert it.
    # Keep the execution serial for consistent, idempotent writes.
    for site in sites:
        process_site(client, lookback_minutes, site)

def process_site(client, lookback_minutes, site):
    oee_space = "oee_ts_space"
    source_space = "icapi_dm_space"

    timeseries = get_time_series_for_site(client, site, source_space)
    if not timeseries:
        return

    asset_eids = sorted({item.external_id.split(sep=":", maxsplit=1)[0] for item in timeseries})
    if not asset_eids:
        return

    instance_ids = [NodeId(space=source_space, external_id=ts.external_id) for ts in timeseries]
    all_latest_dps = client.time_series.data.retrieve_latest(instance_id=instance_ids)
    if not all_latest_dps:
        print(f"No datapoints found for {site}; skipping OEE calculation.")
        return

    latest_by_external_id = {
        latest_dp.instance_id.external_id: latest_dp
        for latest_dp in all_latest_dps
        if latest_dp.instance_id is not None
    }

    for asset in asset_eids:
        input_external_ids = {
            suffix: f"{asset}:{suffix}"
            for suffix in ("count", "good", "status", "planned_status")
        }
        latest_dps = [latest_by_external_id.get(external_id) for external_id in input_external_ids.values()]
        missing_series = [
            external_id
            for external_id, latest_dp in zip(input_external_ids.values(), latest_dps)
            if latest_dp is None or not latest_dp.timestamp
        ]
        if missing_series:
            print(
                f"Skipping {asset}: missing latest datapoints for required source series "
                f"{', '.join(missing_series)}."
            )
            continue

        print(f"Calculating OEE for {asset}")
        input_instance_ids = {
            suffix: NodeId(space=source_space, external_id=external_id)
            for suffix, external_id in input_external_ids.items()
        }
        input_column_names = {
            suffix: f"NodeId({node_id.space}, {node_id.external_id})"
            for suffix, node_id in input_instance_ids.items()
        }
        count_node = input_column_names["count"]
        good_node = input_column_names["good"]
        status_node = input_column_names["status"]
        planned_status_node = input_column_names["planned_status"]

        end = min(dp.timestamp[0] for dp in latest_dps)

        start = end - int(lookback_minutes)
        dps_df = client.time_series.data.retrieve_dataframe(
            instance_id=list(input_instance_ids.values()),
            start=start,
            end=end,
            aggregates=["sum"],
            granularity="1m",
            include_aggregate_name=False,
            limit=None
        )

        if dps_df.empty:
            print(f"No datapoints retrieved for {asset} between {start} and {end}.")
            continue

        # Frontfill because "planned_status" and "status" only have datapoints when the value changes
        dps_df = dps_df.ffill()

        for required_node in input_column_names.values():
            if required_node not in dps_df.columns:
                print(
                    f"Skipping {asset}: dataframe is missing required source series "
                    f"{required_node}. Returned columns: {list(dps_df.columns)}"
                )
                break

            series = dps_df[required_node]
            if required_node in (planned_status_node, status_node):
                first_valid_index = series.first_valid_index()
                if first_valid_index is None:
                    print(f"Skipping {asset}: no values retrieved for required state series {required_node}.")
                    break

                first_valid_value = series.loc[first_valid_index]
                backfill_value = 1.0 if first_valid_value == 0.0 else 0.0
                dps_df[required_node] = series.fillna(value=backfill_value)
        else:
            count_dps = dps_df[count_node]
            good_dps = dps_df[good_node]
            status_dps = dps_df[status_node]
            planned_status_dps = dps_df[planned_status_node]

            total_items = len(count_dps)

            if (
                total_items != len(good_dps)
                or total_items != len(status_dps)
                or total_items != len(planned_status_dps)
            ):
                # We expect ALL dependent timeseries to have the exact same number of datapoints
                # for the specified time range for the calculation to execute.
                print(
                    f"""{asset}: Unable to retrieve datapoints for all required OEE timeseries (count, good, status, planned_status)
                    between {start} and {end}. Ensure that data is available for the time range specified."""
                )
                continue

            # Calculate the components of OEE
            off_spec_node = f"{asset}:off_spec"
            quality_node = f"{asset}:quality"
            performance_node = f"{asset}:performance"
            availability_node = f"{asset}:availability"
            oee_node = f"{asset}:oee"

            dps_df[off_spec_node] = count_dps - good_dps
            dps_df[quality_node] = good_dps / count_dps
            dps_df[performance_node] = (count_dps / status_dps) / (60.0 / 3.0)
            dps_df[availability_node] = status_dps / planned_status_dps

            dps_df[oee_node] = dps_df[quality_node] * dps_df[performance_node] * dps_df[availability_node]

            # Fill in the divide by zeros
            dps_df = dps_df.fillna(value=0.0)
            dps_df = dps_df.replace([np.inf, -np.inf], 0.0)

            # Drop input timeseries
            dps_df = dps_df.drop(columns=[count_node, good_node, status_node, planned_status_node])

            to_insert = [
                {
                    "instance_id": NodeId(space="oee_ts_space", external_id=external_id),
                    "datapoints": list(zip(dps_df[external_id].index, dps_df[external_id]))
                }
                for external_id in dps_df.columns
            ]

            try:
                client.time_series.data.insert_multiple(to_insert)
            except CogniteNotFoundError as e:
                # Create the missing oee timeseries since they don't exist
                ts_to_create = []
                for node_id in e.not_found:
                    print(f"Creating CogniteTimeSeries {node_id}")

                    external_id = node_id["instanceId"]["externalId"]

                    # change external_id to a readable name
                    # Ex: "OSLPROFILTRASYS185:off_spec" to "OSLPROFILTRASYS185 Off Spec"
                    name = external_id.split(":")
                    name[-1] = name[-1].replace("_", " ").title()

                    ts_to_create.append(
                        CogniteTimeSeriesApply(
                            space=oee_space,
                            external_id=external_id,
                            name=" ".join(name),
                            is_step=False,
                            time_series_type="numeric",
                        )
                    )

                client.data_modeling.instances.apply(ts_to_create)
                client.time_series.data.insert_multiple(to_insert)
