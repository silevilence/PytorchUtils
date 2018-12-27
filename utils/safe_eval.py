# ----------------------------------------------------------------------
# I, Babar K. Zafar, the author or of this code dedicate any and all
# copyright interest in this code to the public domain. I make this
# dedication for the benefit of the public at large and to the
# detriment of our heirs and successors. I intend this dedication to
# be an overt act of relinquishment in perpetuity of all present and
# future rights this code under copyright law.
#
# Version 0.1 / May 27 2006
# ----------------------------------------------------------------------

import unittest
import inspect
import ast
import _thread as thread
import time
import builtins

# ----------------------------------------------------------------------
# Module globals.
# ----------------------------------------------------------------------

# Toggle module level debugging mode.
DEBUG = False

# List of all AST node classes in compiler/ast.py.
all_ast_nodes = \
    [name for name, obj in inspect.getmembers(ast)
     if inspect.isclass(obj) and issubclass(obj, ast.AST)]
# print(all_ast_nodes)
# List of all builtin functions and types (ignoring exception classes).
all_builtins = \
    [name for name, obj in inspect.getmembers(builtins)
     if (not inspect.isclass(obj)) or (not issubclass(obj, Exception))]


# print(all_builtins)


# ----------------------------------------------------------------------
# Utilties.
# ----------------------------------------------------------------------


def classname(obj):
    return obj.__class__.__name__


def is_valid_ast_node(name):
    return name in all_ast_nodes


def is_valid_builtin(name):
    return name in all_builtins


def get_node_lineno(node):
    return node.lineno and node.lineno or 0


# ----------------------------------------------------------------------
# Restricted AST nodes & builtins.
# ----------------------------------------------------------------------


# Deny evaluation of code if the AST contain any of the following nodes:
unallowed_ast_nodes = [
    #   'Add', 'And',
    #   'AssAttr', 'AssList', 'AssName', 'AssTuple',
    #   'Assert', 'Assign', 'AugAssign',
    # 'Backquote',
    #   'Bitand', 'Bitor', 'Bitxor', 'Break',
    #   'CallFunc', 'Class', 'Compare', 'Const', 'Continue',
    #   'Decorators', 'Dict', 'Discard', 'Div',
    #   'Ellipsis', 'EmptyNode',
    # 'Exec',
    'ExceptHandler',
    #   'Expression', 'FloorDiv',
    #   'For',
    # 'From',
    #   'Function',
    #   'GenExpr', 'GenExprFor', 'GenExprIf', 'GenExprInner',
    #   'Getattr', 'Global', 'If',
    'Import', 'ImportFrom',
    #   'Invert',
    #   'Keyword', 'Lambda', 'LeftShift',
    #   'List', 'ListComp', 'ListCompFor', 'ListCompIf', 'Mod',
    #   'Module',
    #   'Mul', 'Name', 'Node', 'Not', 'Or', 'Pass', 'Power',
    #   'Print', 'Printnl',
    'Raise',
    #    'Return', 'RightShift', 'Slice', 'Sliceobj',
    #   'Stmt', 'Sub', 'Subscript',
    'Try',
    # 'TryExcept', 'TryFinally',
    #   'Tuple', 'UnaryAdd', 'UnarySub',
    #   'While','Yield'
]

# Deny evaluation of code if it tries to access any of the following builtins:
unallowed_builtins = [
    '__import__',
    #   'abs', 'apply', 'basestring', 'bool', 'buffer',
    #   'callable', 'chr', 'classmethod', 'cmp', 'coerce',
    'compile',
    #   'complex',
    'delattr',
    #   'dict',
    'dir',
    #   'divmod', 'enumerate',
    'eval',  # 'execfile', 'file',
    #   'filter', 'float', 'frozenset',
    'getattr', 'globals', 'hasattr',
    #    'hash', 'hex', 'id',
    'input',
    #   'int', 'intern', 'isinstance', 'issubclass', 'iter',
    #   'len', 'list',
    'locals',
    #   'long', 'map', 'max', 'min', 'object', 'oct',
    'open',
    #   'ord', 'pow', 'property', 'range',
    # 'raw_input',
    #   'reduce',
    # 'reload',
    #   'repr', 'reversed', 'round', 'set',
    'setattr',
    #   'slice', 'sorted', 'staticmethod',  'str', 'sum', 'super',
    #   'tuple', 'type', 'unichr', 'unicode',
    'vars',
    #    'xrange', 'zip'
]

for ast_name in unallowed_ast_nodes:
    # print(ast_name)
    assert (is_valid_ast_node(ast_name))
for name in unallowed_builtins:
    # print(name)
    assert (is_valid_builtin(name))


def is_unallowed_ast_node(kind):
    return kind in unallowed_ast_nodes


def is_unallowed_builtin(name):
    return name in unallowed_builtins


# ----------------------------------------------------------------------
# Restricted attributes.
# ----------------------------------------------------------------------


# In addition to these we deny access to all lowlevel attrs (__xxx__).
unallowed_attr = [
    'im_class', 'im_func', 'im_self',
    'func_code', 'func_defaults', 'func_globals', 'func_name',
    'tb_frame', 'tb_next',
    'f_back', 'f_builtins', 'f_code', 'f_exc_traceback',
    'f_exc_type', 'f_exc_value', 'f_globals', 'f_locals']


def is_unallowed_attr(name):
    return (name[:2] == '__' and name[-2:] == '__') or \
           (name in unallowed_attr)


# ----------------------------------------------------------------------
# SafeEvalVisitor.
# ----------------------------------------------------------------------


class SafeEvalError(Exception):
    """
    Base class for all which occur while walking the AST.

    Attributes:
      errmsg = short decription about the nature of the error
      lineno = line offset to where error occured in source code
    """

    def __init__(self, errmsg, lineno):
        self.errmsg, self.lineno = errmsg, lineno

    def __str__(self):
        return "line %d : %s" % (self.lineno, self.errmsg)

    def __repr__(self):
        return self.__str__()


class SafeEvalASTNodeError(SafeEvalError):
    """Expression/statement in AST evaluates to a restricted AST node type."""
    pass


class SafeEvalBuiltinError(SafeEvalError):
    """Expression/statement in tried to access a restricted builtin."""
    pass


class SafeEvalAttrError(SafeEvalError):
    """Expression/statement in tried to access a restricted attribute."""
    pass


# noinspection PyMethodMayBeStatic
class SafeVisitor(ast.NodeVisitor):
    def __init__(self):
        """Initialize visitor by generating callbacks for all AST node types."""
        self.errors = []
        # noinspection PyShadowingNames
        for ast_name in all_ast_nodes:
            # Don't reset any overridden callbacks.
            if getattr(self, 'visit_' + ast_name, None):
                continue
            if is_unallowed_ast_node(ast_name):
                setattr(self, 'visit_' + ast_name, self.fail)
            # else:
            #     setattr(self, 'visit_' + ast_name, self.ok)

    def visit_Name(self, node):
        """Disallow any attempts to access a restricted builtin/attr."""
        name = node.id
        lineno = get_node_lineno(node)
        if is_unallowed_builtin(name):
            raise (SafeEvalBuiltinError(
                "access to builtin '%s' is denied" % name, lineno))
        elif is_unallowed_attr(name):
            raise (SafeEvalAttrError(
                "access to attribute '%s' is denied" % name, lineno))

        super(SafeVisitor, self).generic_visit(node)

    def visit_Attribute(self, node):
        """Disallow any attempts to access a restricted attribute."""
        name = node.attr
        lineno = get_node_lineno(node)
        if is_unallowed_attr(name):
            raise (SafeEvalAttrError(
                "access to attribute '%s' is denied" % name, lineno))

        super().generic_visit(node)

    def fail(self, node):
        """Default callback for unallowed AST nodes."""
        lineno = get_node_lineno(node)
        raise (SafeEvalASTNodeError(
            "execution of '%s' statements is denied" % classname(node),
            lineno))

    def walk(self, tree):
        try:
            self.visit(tree)
        except SafeEvalError as e:
            return False, e
        return True, None


# ----------------------------------------------------------------------
# Safe 'eval' replacement.
# ----------------------------------------------------------------------


class SafeEvalException(Exception):
    """Base class for all safe-eval related errors."""
    pass


class SafeEvalCodeException(SafeEvalException):
    """
    Exception class for reporting all errors which occured while
    validating AST for source code in safe_eval().

    Attributes:
      code   = raw source code which failed to validate
      errors = list of SafeEvalError
    """

    def __init__(self, code, errors):
        self.code, self.errors = code, errors

    def __str__(self):
        return '\n'.join([str(err) for err in self.errors])

    def __repr__(self):
        return self.__str__()


class SafeEvalContextException(SafeEvalException):
    """
    Exception class for reporting unallowed objects found in the dict
    intended to be used as the local enviroment in safe_eval().

    Attributes:
      keys   = list of keys of the unallowed objects
      errors = list of strings describing the nature of the error
               for each key in 'keys'
    """

    def __init__(self, keys, errors):
        self.keys, self.errors = keys, errors

    def __str__(self):
        return '\n'.join([str(err) for err in self.errors])


class SafeEvalTimeoutException(SafeEvalException):
    """
    Exception class for reporting that code evaluation execeeded
    the given timelimit.

    Attributes:
      timeout = time limit in seconds
    """

    def __init__(self, timeout):
        self.timeout = timeout

    def __str__(self):
        return "Timeout limit execeeded (%s secs) during exec" % self.timeout


def eval_timed(code, context, timeout_secs, executer=eval):
    """
    Dynamically execute 'code' using 'context' as the global enviroment.
    SafeEvalTimeoutException is raised if execution does not finish within
    the given timelimit.
    """
    assert (timeout_secs > 0)

    signal_finished = False

    def alarm(secs):
        # noinspection PyUnusedLocal,PyShadowingNames
        def wait(secs):
            for n in range(timeout_secs):
                time.sleep(1)
                if signal_finished:
                    break
            else:
                thread.interrupt_main()

        thread.start_new_thread(wait, (secs,))

    try:
        alarm(timeout_secs)
        result = executer(code, context)
        signal_finished = True
        return result
    except KeyboardInterrupt:
        raise SafeEvalTimeoutException(timeout_secs)


def safe_eval(code, context=None, timeout_secs=5, executer=eval):
    """
    Validate source code and make sure it contains no unauthorized
    expression/statements as configured via 'unallowed_ast_nodes' and
    'unallowed_builtins'. By default this means that code is not
    allowed import modules or access dangerous builtins like 'open' or
    'eval'. If code is considered 'safe' it will be executed via
    'exec' using 'context' as the global environment. More details on
    how code is executed can be found in the Python Reference Manual
    section 6.14 (ignore the remark on '__builtins__'). The 'context'
    enviroment is also validated and is not allowed to contain modules
    or builtins. The following exception will be raised on errors:

      if 'context' contains unallowed objects = 
        SafeEvalContextException

      if code is didn't validate and is considered 'unsafe' = 
        SafeEvalCodeException

      if code did not execute within the given timelimit =
        SafeEvalTimeoutException
    """
    if context is None:
        context = {}
    ctx_errkeys, ctx_errors = [], []
    for (key, obj) in context.items():
        if inspect.isbuiltin(obj):
            ctx_errkeys.append(key)
            ctx_errors.append("key '%s' : unallowed builtin %s" % (key, obj))
        if inspect.ismodule(obj):
            ctx_errkeys.append(key)
            ctx_errors.append("key '%s' : unallowed module %s" % (key, obj))

    if ctx_errors:
        raise SafeEvalContextException(ctx_errkeys, ctx_errors)

    asttree = ast.parse(code)
    checker = SafeVisitor()

    successful, err = checker.walk(asttree)
    if successful:
        return eval_timed(code, context, timeout_secs, executer)
    else:
        raise SafeEvalCodeException(code, [err])


# ----------------------------------------------------------------------
# Basic tests.
# ----------------------------------------------------------------------


class TestSafeEval(unittest.TestCase):
    def test_builtin(self):
        # attempt to access a unsafe builtin
        self.assertRaises(SafeEvalException,
                          safe_eval, "open('test.txt', 'w')")

    def test_getattr(self):
        # attempt to get arround direct attr access
        self.assertRaises(SafeEvalException,
                          safe_eval, "getattr(int, '__abs__')")

    def test_func_globals(self):
        # attempt to access global enviroment where fun was defined
        self.assertRaises(SafeEvalException,
                          safe_eval, "def x(): pass; print(x.func_globals)")

    def test_lowlevel(self):
        # lowlevel tricks to access 'object'
        self.assertRaises(SafeEvalException,
                          safe_eval, "().__class__.mro()[1].__subclasses__()")

    def test_timeout_ok(self):
        # attempt to exectute 'slow' code which finishes within timelimit
        def test(): time.sleep(2)

        env = {'test': test}
        safe_eval("test()", env, timeout_secs=5)

    def test_timeout_exceed(self):
        # attempt to exectute code which never teminates
        self.assertRaises(SafeEvalException,
                          safe_eval, "while True: pass", None, 5, exec)

    def test_invalid_context(self):
        # can't pass an enviroment with modules or builtins
        env = {'f': builtins.open, 'g': time}
        self.assertRaises(SafeEvalException,
                          safe_eval, "print 1", env)

    def test_callback(self):
        # modify local variable via callback
        self.value = 0

        def test(): self.value = 1

        env = {'test': test}
        safe_eval("test()", env)
        self.assertEqual(self.value, 1)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------------------------------------
# The End.
# ----------------------------------------------------------------------
