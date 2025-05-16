"""
CifFileWriter and MMCIF2Dict from the pdbecif package are used in allodb

However, for homogenicity and to ease the work with Cif data dicts, the _parseFile private method of MMCIF2Dict is modified to always return dicts where the values are always a list, even if it only contains one item, so that the dictionary can directly be used to construct a DataFrame (edited lines w.r.t. the original _parseFile method are highlighted with ###########) 
"""

from pdbecif.mmcif_io import CifFileWriter

from pdbecif.mmcif_tools import * # Includes class MMCIF2Dict with private method _parseFile

def _parseFile(
    self, file_path, ignoreCategories, preserve_token_order, onlyCategories
):
    """Private method that will do the work of parsing the mmCIF data file
    return Dictionary"""

    if preserve_token_order:
        try:
            from collections import OrderedDict as _dict
        except ImportError:
            # fallback: try to use the ordereddict backport when using python 2.6
            try:
                from ordereddict import OrderedDict as _dict
            except ImportError:
                # backport not installed: use local OrderedDict
                from mmCif.ordereddict import OrderedDict as _dict
    else:
        _dict = dict

    mmcif_like_file = _dict()
    data_block = _dict()
    save_block = _dict()

    data_heading = ""
    line_num = 0
    try:
        with openGzip(file_path, "rt") as f1:
            table_names = []
            table_values = []
            table_values_array = []
            isLoop = False
            multiLineValue = False
            skipCategory = False
            for line in f1:
                line_num += 1
                if skipCategory:
                    flag = False
                    while line:
                        check = (
                            line.strip().startswith("_")
                            or self.loopRE.match(line.strip()[:5])
                            or self.saveRE.match(line.strip()[:5])
                            or self.dataRE.match(line.strip()[:5])
                        )
                        if flag:
                            if check:
                                isLoop = False
                                break
                        else:
                            if not check:
                                flag = True
                        if not (
                            self.saveRE.match(line.strip()[:5])
                            or self.dataRE.match(line.strip()[:5])
                        ):
                            try:
                                line = next(f1)
                                line_num += 1
                            except StopIteration:
                                break
                        else:
                            break
                    skipCategory = False

                if (
                    isLoop is True
                    and table_values_array != []
                    and (
                        self.loopRE.match(line) is not None
                        or (line.strip().startswith("_"))
                    )
                ):
                    isLoop = False
                    num_item = len(table_names)
                    if len(table_values_array) % num_item != 0:
                        raise MMCIFWrapperSyntaxError(category)
                    for val_index, item in enumerate(table_names):
                        data_block[category][item] = table_values_array[
                            val_index::num_item
                        ]
                    table_values_array = []

                if line.strip() == "":
                    continue
                if line.startswith("#"):
                    continue
                if "\t#" in line or " #" in line and not line.startswith(";"):
                    new_line = ""
                    for tok in self.dataValueRE.findall(line):
                        if not tok.startswith("#"):
                            new_line += tok + " "
                        else:
                            break
                    # make sure to preserve the fact that ';' was not the first character
                    line = (
                        new_line if not new_line.startswith(";") else " " + new_line
                    )
                    # Fails for entries "3snv", "1kmm", "1ser", "2prg", "3oqd"
                    # line = re.sub(r'\s#.*$', '', line)
                if line.startswith(";"):
                    while "\n;" not in line:
                        try:
                            line += next(f1)
                            line_num += 1
                        except StopIteration:
                            break
                    multiLineValue = True
                if self.dataRE.match(line):
                    if data_block != {}:
                        if table_values_array != []:
                            isLoop = False
                            num_item = len(table_names)
                            if len(table_values_array) % num_item != 0:
                                raise mmCifSyntaxError(category)
                            for val_index, item in enumerate(table_names):
                                data_block[category][item] = table_values_array[
                                    val_index::num_item
                                ]
                            table_names = []
                            table_values_array = []
                        mmcif_like_file[data_heading] = data_block
                        data_block = _dict()
                    data_heading = self.dataRE.match(line).group("data_heading")
                elif self.saveRE.match(line):
                    while line.strip() != "save_":
                        try:
                            line = next(f1)
                            line_num += 1
                        except StopIteration:
                            break
                    continue
                elif self.loopRE.match(line):
                    # Save and clear the table_values_array buffer from the
                    # previous loop that was read
                    if table_values_array != []:
                        for itemIndex, name in enumerate(table_names):
                            data_block[category].update(
                                {
                                    name: [
                                        row[itemIndex] for row in table_values_array
                                    ]
                                }
                            )
                        table_values_array = []
                    isLoop = True
                    category, item, value = None, None, None
                    # Stores items of a category listed in loop blocks
                    table_names = []
                    # Stores values of items in a loop as a single row
                    table_values = []
                elif self.dataNameRE.match(line):
                    # Match category and item simultaneously
                    m = self.dataNameRE.match(line)
                    category = m.group("data_category")
                    item = m.group("category_item")
                    remainder = m.group("remainder")
                    value = None
                    if isLoop and remainder != "":
                        """Append any data values following the last loop
                        category.item tag should any exist"""
                        table_values += self._tokenizeData(remainder)
                        line = ""
                    else:
                        line = remainder + "\n"
                    if not isLoop:
                        if line.strip() != "":
                            value = self._tokenizeData(line)
                        else:
                            # For cases where values are on the following
                            # line
                            try:
                                line = next(f1)
                                line_num += 1
                            except StopIteration:
                                break
                        while value is None:
                            char_start = 1 if line.startswith(";") else 0
                            while line.startswith(
                                ";"
                            ) and not line.rstrip().endswith("\n;"):
                                try:
                                    line += next(f1)
                                    line_num += 1
                                except StopIteration:
                                    break
                            value = (line[char_start : line.rfind("\n;")]).strip()
                            if char_start > 0:
                                value = (
                                    line[char_start : line.rfind("\n;")]
                                ).strip()
                            else:
                                value = self._tokenizeData(" " + line)
                        if (ignoreCategories and category in ignoreCategories) or (
                            onlyCategories and category not in onlyCategories
                        ):
                            pass
                        else:
                            if category in data_block:
                                data_block[category].update(
                                    {item: value} ################################### changed so that it always returns lists
                                )
                            else:
                                data_block.setdefault(
                                    category,
                                    _dict(
                                        {item: value} ############################### changed so that it always returns lists
                                    ),
                                )  # OrderedDict here preserves item order
                    else:
                        if (ignoreCategories and category in ignoreCategories) or (
                            onlyCategories and category not in onlyCategories
                        ):
                            skipCategory = True
                        else:
                            data_block.setdefault(
                                category, _dict()
                            )  # OrderedDict here preserves item order
                            table_names.append(item)
                else:
                    if multiLineValue is True:
                        table_values.append((line[1 : line.rfind("\n;")]).strip())
                        multiLineValue = False
                        line = line[line.rfind("\n;") + 2 :]
                        if line.strip() != "":
                            table_values += self._tokenizeData(line)
                    else:
                        table_values += self._tokenizeData(line)

                    if table_values != []:
                        table_values_array += table_values
                        table_values = []
            if isLoop is True and table_values_array != []:
                isLoop = False
                num_item = len(table_names)
                for val_index, item in enumerate(table_names):
                    data_block[category][item] = table_values_array[
                        val_index::num_item
                    ]
                table_values_array = []
            if data_block != {}:
                mmcif_like_file[data_heading] = data_block
        return mmcif_like_file
    except KeyError as key_err:
        print("KeyError [line %i]: %s" % (line_num, str(key_err)))
    except IOError as io_err:
        print("IOException [line %i]: %s" % (line_num, str(io_err)))



MMCIF2Dict._parseFile = _parseFile