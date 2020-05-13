import asyncio
import sys
from typing import Any, Dict

import httpx

from ..requests.api_dataset import APIDataSet


def asyncio_run(aw):
    if sys.version_info >= (3, 7):
        return asyncio.run(aw)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        r = loop.run_until_complete(aw)
    finally:
        loop.close()
        asyncio.set_event_loop(None)

    return r


async def request_coroutine(session, method, url, request_args):
    return await session.request(method, url=url, **request_args)


async def requests_coroutine(session_config, method, url_list, request_args):
    async with httpx.AsyncClient(**session_config) as session:
        request_coroutines = [
            request_coroutine(session, method, url, request_args) for url in url_list
        ]

        request_tasks = [
            asyncio.ensure_future(coroutine) for coroutine in request_coroutines
        ]
        r = await asyncio.wait(request_tasks)
    return r


class AsyncAPIDataSet(APIDataSet):
    def _configure_session(self, session_config, _):
        return session_config

    def _execute_request(self) -> Dict[str, Any]:

        request_args = self._request_args
        method = self._method
        url_dict = self._get_url_dict()
        session_config = self._session

        name_url_list = list(url_dict.items())
        url_list = [e[1] for e in name_url_list]
        tasks_done, tasks_pending = asyncio_run(
            requests_coroutine(session_config, method, url_list, request_args)
        )

        name_list = [e[0] for e in name_url_list]
        response_dict = {}
        for name, task in zip(name_list, tasks_done):
            try:
                response_dict[name] = task.result()
            except Exception as exc:
                response_dict[name] = self._handle_exceptions(exc)

        return response_dict
