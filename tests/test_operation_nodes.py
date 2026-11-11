"""ComfyUI coverage and typed catalog execution."""
from pathlib import Path
from remove_ai_watermarks.integration import operation_catalog
import operation_nodes


def test_all_operations_get_nodes_and_all_parameters_get_controls():
    classes, names = operation_nodes.node_mappings()
    assert set(classes) == {'RAIWOperation_' + item['id'] for item in operation_catalog()}
    for node in classes.values():
        assert set(node.INPUT_TYPES()['required']) == {p['name'] for p in node.operation['parameters']}
    assert set(classes) == set(names)


def test_real_text_operation_and_file_result_preserve_the_input(tmp_path):
    source = tmp_path / 'source.txt'
    source.write_text('My own text.\u200b\u200b\u200b\u200b')
    before = source.read_bytes()
    classes, _ = operation_nodes.node_mappings()
    node = classes['RAIWOperation_cli_text_metadata']()
    output, report, exit_code = node.execute(source=str(source), remove=True)
    assert exit_code == 0, report
    assert Path(output).is_file()
    assert source.read_bytes() == before


def test_fixed_library_seeds_do_not_enable_comfy_auto_randomization():
    classes, _ = operation_nodes.node_mappings()
    video = classes['RAIWOperation_cli_video_invisible'].INPUT_TYPES()['required']
    assert video['seed'][1].get('control_after_generate') is False


def test_file_operations_are_not_reused_from_workflow_cache():
    import math
    classes, _ = operation_nodes.node_mappings()
    assert math.isnan(classes['RAIWOperation_cli_text_identify'].IS_CHANGED(source='same-path.txt'))
