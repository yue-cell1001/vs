import csv
import tempfile
from pathlib import Path
from math import isfinite
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED
from .plot import choose_entry
from .schedule import write_tables
NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'

def _column(number):
    label = ''
    while number:
        number, remainder = divmod(number - 1, 26)
        label = chr(65 + remainder) + label
    return label

def _sheet(rows):
    yield f'<worksheet xmlns="{NS}"><sheetData>'
    for index, row in enumerate(rows, 1):
        yield f'<row r="{index}">'
        for column, value in enumerate(row, 1):
            if isinstance(value, (int, float)) and (not isinstance(value, bool)) and isfinite(value):
                yield f'<c r="{_column(column)}{index}"><v>{value}</v></c>'
                continue
            text = ''.join((c for c in str(value) if ord(c) >= 32 or c in '\t\n\r'))
            yield f'<c r="{_column(column)}{index}" t="inlineStr"><is><t xml:space="preserve">{escape(text)}</t></is></c>'
        yield '</row>'
    yield '</sheetData></worksheet>'

def export_workbook(solutions, net, p, directory, stats, solution_name=None):
    entries = [{'solution': f'solution_{i:03}', 'F1': s.objectives[0], 'F2': s.objectives[1]} for i, s in enumerate(solutions)]
    selected, selection = choose_entry(entries, solution_name)
    chosen = solutions[int(selected['solution'].split('_')[-1])]
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'final_dispatch.xlsx'
    temporary = root / 'final_dispatch.xlsx.tmp'
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        write_tables(chosen, net, p, folder, 'final')
        sheets = [('summary', [['key', 'value'], ['solution', selected['solution']], ['selection_method', selection['method']], ['F1', selected['F1']], ['F2', selected['F2']], ['timetable_number', chosen.metadata.get('timetable_index', -1) + 1], ['computation_seconds', sum((s.get('elapsed_seconds', 0) for s in stats))]]), ('pareto', [['solution', 'F1', 'F2']] + [[e['solution'], e['F1'], e['F2']] for e in entries]), ('computation_time', [['timetable', 'elapsed_seconds', 'iterations', 'complete_feasible_iterations']] + [[s['timetable'] + 1, s.get('elapsed_seconds', ''), s.get('iterations', ''), s.get('complete_feasible_iterations', '')] for s in stats]), ('iteration_time', [['timetable', 'iteration', 'elapsed_seconds', 'repair_complete']] + [[s['timetable'] + 1, t['iteration'], t.get('elapsed_seconds', ''), t.get('repair_complete', False)] for s in stats for t in s.get('trace', [])])]
        for file in sorted(folder.glob('final_*.csv')):
            with file.open(encoding='utf-8-sig', newline='') as stream:
                sheets.append((file.stem.removeprefix('final_'), list(csv.reader(stream))))
        with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
            overrides = ''.join((f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1, len(sheets) + 1)))
            archive.writestr('[Content_Types].xml', '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>' + overrides + '</Types>')
            archive.writestr('_rels/.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
            archive.writestr('xl/workbook.xml', f'<workbook xmlns="{NS}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>' + ''.join((f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>' for i, (name, _) in enumerate(sheets, 1))) + '</sheets></workbook>')
            archive.writestr('xl/_rels/workbook.xml.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + ''.join((f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(sheets) + 1))) + '</Relationships>')
            for i, (_, rows) in enumerate(sheets, 1):
                archive.writestr(f'xl/worksheets/sheet{i}.xml', ''.join(_sheet(rows)))
    temporary.replace(target)
    return target
