# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""``APIDataSet`` loads the data from HTTP(S) APIs
and returns them into either as string or json Dict.
It uses the python requests library: https://requests.readthedocs.io/en/master/
"""
import socket
from typing import Any, Dict, List, Tuple, Union

import requests
from requests.auth import AuthBase

from ..core import AbstractDataSet, DataSetError


class APIDataSet(AbstractDataSet):
    """``APIDataSet`` loads the data from HTTP(S) APIs.
    It uses the python requests library: https://requests.readthedocs.io/en/master/

    Example:
    ::

        >>> from kedro.extras.datasets.api import APIDataSet
        >>>
        >>>
        >>> data_set = APIDataSet(
        >>>     url="https://quickstats.nass.usda.gov"
        >>>     params={
        >>>         "key": "SOME_TOKEN",
        >>>         "format": "JSON",
        >>>         "commodity_desc": "CORN",
        >>>         "statisticcat_des": "YIELD",
        >>>         "agg_level_desc": "STATE",
        >>>         "year": 2000
        >>>     }
        >>> )
        >>> data = data_set.load()

    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        url: Union[str, List[str], Dict[str, str]] = None,
        method: str = "GET",
        data: Any = None,
        params: Dict[str, Any] = None,
        headers: Dict[str, Any] = None,
        auth: Union[Tuple[str], AuthBase] = None,
        timeout: int = 60,
        attribute: str = "",
        skip_errors: bool = False,
        transforms: List[callable] = [],
        session_config: Dict[str, Any] = {},
        pool_config: Dict[str, Dict[str, Any]] = {
            "https://": {
                "pool_connections": 10,
                "pool_maxsize": 10,
                "max_retries": 0,
                "pool_block": False,
            },
            "http://": {
                "pool_connections": 10,
                "pool_maxsize": 10,
                "max_retries": 0,
                "pool_block": False,
            },
        },
    ) -> None:
        """Creates a new instance of ``APIDataSet`` to fetch data from an API endpoint.

        Args:
            url: The API URL endpoint.
            method: The Method of the request, GET, POST, PUT, DELETE, HEAD, etc...
            data: The request payload, used for POST, PUT, etc requests
                https://requests.readthedocs.io/en/master/user/quickstart/#more-complicated-post-requests
            params: The url parameters of the API.
                https://requests.readthedocs.io/en/master/user/quickstart/#passing-parameters-in-urls
            headers: The HTTP headers.
                https://requests.readthedocs.io/en/master/user/quickstart/#custom-headers
            auth: Anything ``requests`` accepts. Normally it's either ``('login', 'password')``,
                or ``AuthBase``, ``HTTPBasicAuth`` instance for more complex cases.
            timeout: The wait time in seconds for a response, defaults to 1 minute.
                https://requests.readthedocs.io/en/master/user/quickstart/#timeouts
            attribute: The attribute of response to return. Normally it's either `text`, which
                returns pure text,`json`, which returns JSON in Python Dict format, `content`,
                which returns a raw content, or `` (empty string), which returns the response
                object itself. Defaults to `` (empty string).
            skip_errors: If True, exceptions will not interrupt loading data and be returned
                instead of the expected responses by _load method. Defaults to False.
            transforms: List of callables to transform the output.
            session_config: Dict of arguments fed to the session.
            pool_config: Dict of mounting prefix key to Dict of requests.adapters.HTTPAdapter
                param key to value.
                https://requests.readthedocs.io/en/master/user/advanced/#transport-adapters
                https://urllib3.readthedocs.io/en/latest/advanced-usage.html

        """
        super().__init__()
        self._request_args: Dict[str, Any] = {
            "data": data,
            "params": params,
            "headers": headers,
            "auth": auth,
            "timeout": timeout,
        }

        self._url = url
        self._method = method
        self._attribute = attribute
        self._skip_errors = skip_errors
        self._transforms = transforms

        self._session_config = session_config
        self._pool_config = pool_config
        self._session = self._configure_session(session_config, pool_config)

    def _configure_session(self, session_config, pool_config):
        session = requests.Session(**session_config)
        for prefix, adapter_params in pool_config.items():
            session.mount(prefix, requests.adapters.HTTPAdapter(**adapter_params))

        return session

    def _describe(self) -> Dict[str, Any]:
        return dict(
            **self._request_args,
            url=self._url,
            method=self._method,
            session_config=self._session_config,
            pool_config=self._pool_config,
            attribute=self._attribute,
            skip_errors=self._skip_errors,
        )

    def _get_url_dict(self):
        if isinstance(self._url, str):
            url_dict = {"_": self._url}
        elif isinstance(self._url, list):
            url_dict = {i: url for (i, url) in enumerate(self._url)}
        else:
            url_dict = self._url
        return url_dict

    def _execute_request(self) -> Dict[str, requests.Response]:

        request_args = self._request_args
        session = self._session
        method = self._method
        url_dict = self._get_url_dict()

        def request(url):
            response = session.request(method, url=url, **request_args)
            response.raise_for_status()
            return response

        response_dict = {}
        for name, url in url_dict.items():
            try:
                response_dict[name] = request(url)
            except Exception as exc:
                response_dict[name] = self._handle_exceptions(exc)

        return response_dict

    def _handle_exceptions(self, exc):

        if isinstance(exc, requests.exceptions.HTTPError):
            e = DataSetError("Failed to fetch data", exc)
        elif isinstance(exc, socket.error):
            e = DataSetError("Failed to connect to the remote server")
        else:
            e = DataSetError("Exception", exc)

        if self._skip_errors:
            return e
        raise e

    def _load(self) -> Any:
        response_dict = self._execute_request()

        output_dict = {}
        for name, response in response_dict.items():
            if isinstance(response, Exception):
                output_dict[name] = response
                continue
            if response.status_code != requests.codes.ok:
                output_dict[name] = response
                continue

            if not self._attribute:
                output = response
            elif hasattr(response, self._attribute):
                if self._attribute == "json":
                    output = response.json()
                else:
                    output = getattr(response, self._attribute)
            elif self._skip_errors:
                output_dict[name] = response
                continue
            else:
                raise DataSetError(
                    "Response has no attribute: {}".format(self._attribute)
                )

            try:
                for transform in self._transforms:
                    output = transform(output)
                output_dict[name] = output
            except Exception as exc:
                e = DataSetError("Exception", exc)
                if self._skip_errors:
                    output_dict[name] = e
                    continue
                else:
                    raise e

        if isinstance(self._url, str):
            return next(iter(output_dict.values()))
        elif isinstance(self._url, list):
            return [output_dict[i] for i in range(len(output_dict))]
        else:
            return output_dict

    def _save(self, data: Any) -> None:
        raise DataSetError(
            "{} is a read only data set type".format(self.__class__.__name__)
        )

    def _exists(self) -> bool:
        response_dict = self._execute_request()

        return all(
            [
                getattr(response, "status_code") == requests.codes.ok
                for response in response_dict.values()
            ]
        )

    def __call__(
        self,
        url: Union[str, List[str], Dict[str, str]] = None,
        method: str = None,
        data: Any = None,
        params: Dict[str, Any] = None,
        headers: Dict[str, Any] = None,
        auth: Union[Tuple[str], AuthBase] = None,
        timeout: int = None,
        attribute: str = None,
        skip_errors: bool = None,
        transforms: List[callable] = None,
    ):
        if data is not None:
            self._request_args.update({"data": data})
        if params is not None:
            self._request_args.update({"params": params})
        if headers is not None:
            self._request_args.update({"headers": headers})
        if auth is not None:
            self._request_args.update({"auth": auth})
        if timeout is not None:
            self._request_args.update({"timeout": timeout})
        if url is not None:
            self._url = url
        if method is not None:
            self._method = method
        if attribute is not None:
            self._attribute = attribute
        if skip_errors is not None:
            self._skip_errors = skip_errors
        if transforms is not None:
            self._transforms = transforms
        return self._load()
