class Requirements(object):
    def __init__(self, line_no: int):
        # self.file_name = file_path
        self.line_no = line_no
        self.hit_status = False
        self.id = 0
        self.count = 0

    def is_visited(self, ):
        # self.count += 1
        self.hit_status = True

    def get_lineno(self, ):
        return self.line_no

    def get_hit_status(self):
        return self.hit_status


class LineRequirements(Requirements):
    def __init__(self, line_no: int):
        super(LineRequirements, self).__init__(line_no=line_no)


class BranchRequirements(Requirements):
    def __init__(self, line_no, branch_key):
        super(BranchRequirements, self).__init__(line_no=line_no)
        self.branch_key = branch_key
        self.branch_no = branch_key.split("_")[-1]


class PredicateRequirements(Requirements):
    def __init__(self, line_no, predicate_key):
        super(PredicateRequirements, self).__init__(line_no=line_no)
        self.predicate_key = predicate_key

# Below is Not Used
class PyBranchRequirements(Requirements):
    def __init__(self, line_no, branch_end):
        super(PyBranchRequirements, self).__init__(line_no=line_no)
        self.branch_end = branch_end
        self.if_hit = False
        # Note that if_hit doesn't always mean if branch is hit, because the python profiling phase make the if and else number not stable
        # If the instr is "PUP_JUMP_IF_TRUE", then the self.if_hit represent if branch
        # If the instr is "POP_JUMP_IF_FALSE", then the self.else_hit represent the else branch.....
        # It is one drawback of intermediate representation
        self.else_hit = False


class PyLineRequirements(Requirements):
    def __init__(self, line_no):
        super(PyLineRequirements, self).__init__(line_no=line_no)
        self.stmts = []

    def add(self, stmt):
        self.stmts.append(stmt)


class CLineRequirements(Requirements):
    def __init__(self, line_no, ):
        super(CLineRequirements, self).__init__(line_no=line_no)


class CBranchRequirements(Requirements):
    def __init__(self, line_no, branch_no):
        super(CBranchRequirements, self).__init__(line_no=line_no)
        self.branch_no = branch_no


class NeuronRequirements(Requirements):
    def __init__(self, model_name, layer_no, neuron_no):
        self.id = f"{layer_no}_{neuron_no}"
        super(NeuronRequirements, self).__init__(line_no=self.id)
