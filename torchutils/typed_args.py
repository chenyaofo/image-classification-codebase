'''
Modified from https://github.com/SunDoge/typed-args/tree/v0.4

BSD 3-Clause License

Copyright (c) 2019, SunDoge
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import typing
import collections
import logging
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Union, Optional, Any, Iterable, List, Tuple

try:
    # Python 3.8
    from typing import get_origin, get_args
except ImportError:
    def get_origin(tp):
        """Get the unsubscripted version of a type.
        This supports generic types, Callable, Tuple, Union, Literal, Final and ClassVar.
        Return None for unsupported types. Examples::
            get_origin(Literal[42]) is Literal
            get_origin(int) is None
            get_origin(ClassVar[int]) is ClassVar
            get_origin(Generic) is Generic
            get_origin(Generic[T]) is Generic
            get_origin(Union[T, int]) is Union
            get_origin(List[Tuple[T, T]][int]) == list
        """
        if isinstance(tp, typing._GenericAlias):
            return tp.__origin__
        if tp is typing.Generic:
            return typing.Generic
        return None

    def get_args(tp):
        """Get type arguments with all substitutions performed.
        For unions, basic simplifications used by Union constructor are performed.
        Examples::
            get_args(Dict[str, int]) == (str, int)
            get_args(int) == ()
            get_args(Union[int, Union[T, int], str][int]) == (int, str)
            get_args(Union[int, Tuple[T, int]][str]) == (int, Tuple[str, int])
            get_args(Callable[[], T][int]) == ([], int)
        """
        if isinstance(tp, typing._GenericAlias):
            res = tp.__args__
            if get_origin(tp) is collections.abc.Callable and res[0] is not Ellipsis:
                res = (list(res[:-1]), res[-1])
            return res
        return ()

__version__ = '0.4.0'


@dataclass
class TypedArgs:

    @classmethod
    def from_args(cls, args: Optional[List[str]] = None, namespace: Optional[Namespace] = None):
        typed_args = cls()
        typed_args._parse_args(typed_args.parser_factory(),
                               args=args, namespace=namespace)
        return typed_args

    @classmethod
    def from_known_args(cls, args: Optional[List[str]] = None, namespace: Optional[Namespace] = None):
        typed_args = cls()
        args = typed_args._parse_known_args(
            typed_args.parser_factory(), args=args, namespace=namespace)
        return typed_args, args

    def parser_factory(self):
        return ArgumentParser()

    def _add_arguments(self, parser: ArgumentParser):
        for name, annotation in self.__dataclass_fields__.items():
            self._add_argument(parser, name, annotation.type)

    def _add_argument(self, parser: ArgumentParser, name: str, annotation: Any):
        values: Union[PhantomAction,
                      Tuple[PhantomAction]] = getattr(self, name)

        if type(values) != tuple:
            types = (annotation,)
            values = (values,)
        else:
            """
            List[Union[str, int]]
            """
            types = get_inner_types(annotation)
            if types[-1] == Ellipsis:
                types = (types[0],) * len(values)

        for argument_type, value in zip(types, values):
            if not isinstance(value, PhantomAction):
                continue

            kwargs = value.to_kwargs()
            """
            If there's no option_strings, it must be a position argument.
            For compatible, we get an empty tuple
            """
            args = kwargs.pop('option_strings', ())

            kwargs['dest'] = name  # 必定有dest

            is_position_argument = len(args) == 0

            origin = get_origin(argument_type)

            if origin is list:
                argument_type = get_args(argument_type)[0]

            if kwargs['action'] == 'store':
                if kwargs.get('default') is None:

                    if origin is Union:  # Optional
                        argument_type = get_args(argument_type)[
                            0]  # Get first type
                        kwargs['required'] = False

                    elif origin is None:
                        if not is_position_argument:
                            kwargs['required'] = True

                kwargs['type'] = argument_type

            parser.add_argument(*args, **kwargs)

    def _parse_args(self, parser: ArgumentParser, args: Optional[List[str]] = None,
                    namespace: Optional[Namespace] = None):
        self._add_arguments(parser)
        parsed_args = parser.parse_args(args=args, namespace=namespace)
        self._update_arguments(parsed_args)

    def _parse_known_args(self, parser: ArgumentParser, args: Optional[List[str]] = None,
                          namespace: Optional[Namespace] = None):
        self._add_arguments(parser)
        parsed_args, args = parser.parse_known_args(
            args=args, namespace=namespace)
        self._update_arguments(parsed_args)
        return args

    def _update_arguments(self, parsed_args: Namespace):

        for name, value in parsed_args.__dict__.items():
            setattr(self, name, value)


def get_inner_types(annotation):
    """
    List[Union[int, str]] -> get (int, str)
    """
    return annotation.__args__[0].__args__


@dataclass
class PhantomAction:
    option_strings: Tuple[str, ...]
    action: str = 'store'
    nargs: Union[int, str, None] = None
    const: Any = None
    default: Any = None
    choices: Optional[Iterable] = None
    required: Optional[bool] = None
    help: Optional[str] = None
    metavar: Optional[str] = None

    def to_kwargs(self):
        """
        Follow the argparse.add_argument rules
        """
        kwargs = self.__dict__.copy()

        if len(self.option_strings) == 0:
            # position argument is always required
            del kwargs['required']

        if self.action.startswith('store_'):
            del kwargs['nargs']
            del kwargs['choices']

            if self.action in ['store_true', 'store_false']:
                del kwargs['metavar']
                del kwargs['const']
                del kwargs['default']

        elif self.action == 'append_const':
            del kwargs['nargs']
            del kwargs['choices']

        elif self.action == 'count':
            del kwargs['nargs']
            del kwargs['choices']
            del kwargs['const']
            del kwargs['metavar']

        elif self.action == 'help':
            del kwargs['dest']
            del kwargs['default']

            del kwargs['nargs']
            del kwargs['choices']
            del kwargs['const']
            del kwargs['metavar']
            del kwargs['required']

        elif self.action == 'version':
            del kwargs['dest']
            del kwargs['default']

            del kwargs['nargs']
            del kwargs['choices']
            del kwargs['const']
            del kwargs['metavar']
            del kwargs['required']

        elif self.action == 'parsers':
            raise NotImplementedError()

        return kwargs


def add_argument(
        *option_strings: Union[str, Tuple[str, ...]],
        action: str = 'store',  # in argparse, default action is 'store'
        nargs: Union[int, str, None] = None,
        const: Optional[Any] = None,
        default: Optional[Any] = None,
        choices: Optional[Iterable] = None,
        required: Optional[bool] = None,
        help: Optional[str] = None,
        metavar: Optional[str] = None,
):
    """
    :param option_strings:
    :param action:
    :param nargs:
    :param const:
    :param default:
    :param choices:
    :param required:
    :param help:
    :param metavar:
    :return:
    """
    kwargs = locals()
    return PhantomAction(**kwargs)
