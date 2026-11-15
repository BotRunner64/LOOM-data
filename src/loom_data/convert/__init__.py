"""Convert a source dataset to the export contract."""


def convert(dataset, source, output, **options):
    """Convert in this process and publish `output`; rerun to resume after a failure."""
    from loom_data.convert.config import make_config
    from loom_data.convert.job import run_serial
    run_serial(make_config(dataset, source, **options), output)
