from classes.requirements import PyBranchRequirements, PyLineRequirements, CBranchRequirements, CLineRequirements, LineRequirements, BranchRequirements
import dis
import marshal
import sys
import time
import types
import os
import numpy as np
import collections
from utils.utils import parse_miss_line, concatenate_vector
import io
from contextlib import redirect_stdout


class Files(object):
    def __init__(self, name, path):
        # self.file_name = name  # the name will be the python (.py) file or cpp file (.c,.cc.,cpp,etc)
        self.file_path = path  # the path will be the path of python file
        # profiling on the target file
        # return target self.py_lines: lineRequirements
        self.lines = {}  # define lines and branches as the dict so we can boost the search efficiency in further implementation
        self.branches = {}
        self.total_instruments = []

    @property
    def coverage(self):
        def _coverage(trs):
            num_hit = 0
            num_miss = 0
            for tr in trs:
                if trs[tr].get_hit_status():
                    num_hit += 1
                else:
                    num_miss += 1
            cov = num_hit / (num_hit + num_miss) if (num_hit + num_miss) != 0 else 0
            return num_hit, num_miss, cov

        line_hit, line_miss, line_cov = _coverage(self.lines)
        branch_hit, branch_miss, branch_cov = _coverage(self.branches)
        return line_hit, line_miss, line_cov, branch_hit, branch_miss, branch_cov

    def add_trs(self, lineno=None, branch_key=None, visit=False):
        # lineno: int, branch_key: str
        # add visit function to decide whether the test requirement is visited or not
        if lineno is not None:
            line = LineRequirements(lineno)
            if visit is True:
                line.is_visited()
            self.lines[lineno] = line
        if branch_key is not None:
            line_no = int(branch_key.split("_")[0])
            branch = BranchRequirements(line_no, branch_key)
            if visit is True:
                branch.is_visited()
            self.branches[branch_key] = branch

    def sort_trs(self):
        # sort all line test requirements and branch test requirements based on key
        self.lines = dict(sorted(self.lines.items()))
        self.branches = dict(sorted(self.branches.items()))

    def get_tr_vector(self):
        # get the tr_vector of this file
        # note that we only return the branch tr as the logic vector
        def get_vector(tr_dict):
            vector = np.zeros(len(tr_dict), dtype=np.int)
            for i, tr in enumerate(tr_dict.values()):
                if tr.get_hit_status():  # if this tr is hit
                    vector[i] = 1
            return vector
        # return concatenate_vector(get_vector(self.lines), get_vector(self.branches))
        return get_vector(self.branches)


    def compare_and_update(self, target_file):
        # compare self.lines, compare self.branches
        # Finished: also compare and update the not visited test requirements
        new_line_no = []
        new_branch = []
        # update invoked trs
        for line in target_file.lines.keys():
            if target_file.lines[line].get_hit_status():  # if the tr is hit
                if line not in self.lines.keys():  # if new hit line haven't been recorded yet
                    new_line_no.append(line)
                    self.add_trs(lineno=line, visit=True)
                elif not self.lines[line].get_hit_status():  # if the new line has already been recorded but has not been visited
                    new_line_no.append(line)
                    self.lines[line].is_visited()
            else:  # if the tr hasn't been hit
                if line not in self.lines.keys():  # if the new tr haven't been recorded yet
                    self.add_trs(lineno=line, visit=False)  # record it
        new_line_no.sort()
        for branch_key in target_file.branches.keys():
            lineno, branch_info = branch_key.split("_")
            if target_file.branches[branch_key].get_hit_status():  # if the tr is hit
                if branch_key not in self.branches.keys():  # if the new tr branch haven't been recorded yet
                    new_branch.append(branch_key)
                    self.add_trs(lineno=int(lineno), branch_key=branch_key, visit=True)
                elif not self.branches[branch_key].get_hit_status():  # if the new branch has already been recorded but hasn't been visited
                    new_branch.append(branch_key)
                    self.branches[branch_key].is_visited()
            else:  # if the tr hasn't been hit
                if branch_key not in self.branches.keys():  # if the new tr hasn't been recorded yet
                    self.add_trs(lineno=int(lineno), branch_key=branch_key, visit=False)  # record it
        compare_result = (new_line_no, new_branch)
        return compare_result

    def compare(self, target_file):
        # compare self.lines, compare self.branches
        # Finished: also compare and update the not visited test requirements
        new_line_no = []
        new_branch = []
        # update invoked trs
        for line in target_file.lines.keys():
            if target_file.lines[line].get_hit_status():  # if the tr is hit
                if line not in self.lines.keys():  # if new hit line haven't been recorded yet
                    new_line_no.append(line)
                elif not self.lines[line].get_hit_status():  # if the new line has already been recorded but has not been visited
                    new_line_no.append(line)
        new_line_no.sort()
        for branch_key in target_file.branches.keys():
            if target_file.branches[branch_key].get_hit_status():  # if the tr is hit
                if branch_key not in self.branches.keys():  # if the new tr branch haven't been recorded yet
                    new_branch.append(branch_key)
                elif not self.branches[branch_key].get_hit_status():  # if the new branch has already been recorded but hasn't been visited
                    new_branch.append(branch_key)
        compare_result = (new_line_no, new_branch)
        return compare_result

class CFiles(Files):
    def __init__(self, name, path):
        super(CFiles, self).__init__(name, path)

    # def compare_and_update(self, target_file):
    #     # compare self.lines, compare self.branches
    #     # Finished: also compare and update the not visited test requirements
    #     compare_result = None
    #     new_line_no = []
    #     new_branch = []
    #     # update invoked trs
    #     for line in target_file.lines.keys():
    #         if target_file.lines[line].get_hit_status():  # if the tr is hit
    #             if line not in self.lines.keys():  # if new hit line haven't been recorded yet
    #                 new_line_no.append(line)
    #                 self.add_trs(lineno=line, visit=True)
    #             elif not self.lines[line].get_hit_status():  # if the new line has already been recorded but has not been visited
    #                 new_line_no.append(line)
    #                 self.lines[line].is_visited()
    #         else:  # if the tr hasn't been hit
    #             if line not in self.lines.keys():  # if the new tr haven't been recorded yet
    #                 self.add_trs(lineno=line, visit=False)  # record it
    #     new_line_no.sort()
    #     for branch_key in target_file.branches.keys():
    #         lineno, branch_info = branch_key.split("_")
    #         if target_file.branches[branch_key].get_hit_status():  # if the tr is hit
    #             if branch_key not in self.branches.keys():  # if the new tr branch haven't been recorded yet
    #                 new_branch.append(branch_key)
    #                 self.add_trs(lineno=int(lineno), branch_key=branch_key, visit=True)
    #             elif not self.branches[branch_key].get_hit_status():  # if the new branch has already been recorded but hasn't been visited
    #                 new_branch.append(branch_key)
    #                 self.branches[branch_key].is_visited()
    #         else:  # if the tr hasn't been hit
    #             if branch_key not in self.branches.keys():  # if the new tr hasn't been recorded yet
    #                 self.add_trs(lineno=int(lineno), branch_key=branch_key, visit=False)  # record it
    #     compare_result = (new_line_no, new_branch)
    #     return compare_result


class PyFiles(Files):
    def __init__(self, name, path):
        super(PyFiles, self).__init__(name, path)
        self.name_no_extension = self.file_name.replace(".py", "")
        self.pyc_path = os.path.join(os.path.dirname(self.file_path), "__pycache__",
                                     "{}.cpython-36.pyc".format(self.name_no_extension))
        # self.lines: [pyLineRequirements]
        # self.branches: [pyBranchRequirements]

    def __get_long(self, s):
        return s[0] + (s[1] << 8) + (s[2] << 16) + (s[3] << 24)

    def __collect_trs(self, code):
        f = io.StringIO()
        with redirect_stdout(f):
            dis.disassemble(code)
        bytecode = dis.Bytecode(code)
        line = None
        search_branch_end = 0
        branch_start_line = 0
        for instr in bytecode:
            if instr.starts_line != None:
                if search_branch_end != 0 and instr.offset == search_branch_end:  # we are searching for the end of the branch
                    branch = PyBranchRequirements(self.file_name, branch_start_line, instr.starts_line)
                    self.branches[branch.get_lineno()] = branch  # note that for python branch requirement, each branch requirement represents two test requirements: end_branch and else branch
                    search_branch_end = 0
                if line != None:
                    self.lines[line.get_lineno()] = line  # collect all line test requirements
                    line = PyLineRequirements(self.file_name, instr.starts_line)  # create pyLineRequirements object
                    line.add(instr)  # add statement to line
                elif instr.starts_line is not None and line is None:
                    line = PyLineRequirements(self.file_name, instr.starts_line)
                    line.add(instr)
            else:
                line.add(instr)
            # when this instr have if statement
            if instr.opname in ["POP_JUMP_IF_TRUE", "POP_JUMP_IF_FALSE"] and search_branch_end == 0:
                branch_start_line = line.get_lineno()
                search_branch_end = instr.argval  # get the target node
        self.lines[line.get_lineno()] = line
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                self.__collect_trs(const)
            else:
                pass

    def parse_pyc(self):
        with open(self.pyc_path, 'rb') as f:
            magic_str = f.read(4)
            mtime_str = f.read(4)
            mtime = self.__get_long(mtime_str)
            modtime = time.asctime(time.localtime(mtime))
            source_size = self.__get_long(f.read(4))
            self.__collect_trs(marshal.loads(f.read()))

    def profile(self):
        self.parse_pyc()

    def update_miss_line(self, miss_line):
        file_update_result = None
        max_line = max(self.lines.keys())
        miss_range, miss_branches = parse_miss_line(miss_line, max_line)
        new_line_no = []
        for line in self.lines.values():
            if line.get_hit_status() == False and miss_range[line.get_lineno()] != 0:  # in new log, this line is hit
                line.is_visited()
                new_line_no.append(line.get_lineno())

        new_branch = []
        for branch in self.branches.values():
            branch_no = branch.get_lineno()
            if branch.if_hit and branch.else_hit:  # if two branches of this if condition is hit
                continue
            # at least one branch in this test requirement is missed
            if miss_range[branch_no] == 0:  # if the beginning line of the condition isn't hit
                continue
            if str(branch_no) not in miss_branches.keys():
                # if there is no str(branch_no) key in miss_branches dict, it means that all branches in this condition is invoked
                branch.if_hit = True
                branch.else_hit = True
                new_branch.append(branch.get_lineno())
                continue
            # if branch_no is still in the miss_branch dict, it means that either if branch is missed or else branch is missed
            if len(miss_branches[str(branch_no)]) > 1:
                continue
            else:
                # only one branch is missed, it should be either if or else
                if branch.if_hit == False and str(branch.branch_end) not in miss_branches[str(branch_no)]:
                    # if if branch is not missed, then else branch is missed
                    new_branch.append(branch.get_lineno())
                    branch.if_hit = True
                elif branch.else_hit == False and str(branch.branch_end) in miss_branches[str(branch_no)]:
                    # if if branch is not missed, then else branch is missed
                    branch.else_hit = True
                    new_branch.append(branch.get_lineno())
        file_update_result = (new_line_no, new_branch)
        return file_update_result
