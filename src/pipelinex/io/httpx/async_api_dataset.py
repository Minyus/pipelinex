from typing import Any, Dict
import asyncio
import httpx

from ..requests.api_dataset import APIDataSet


async def coroutine(session, method, url, request_args):
    return await session.request(method, url=url, **request_args)


async def wait_coroutines(coroutines):
    tasks = [asyncio.ensure_future(coroutine) for coroutine in coroutines]
    return await asyncio.wait(tasks)


class AsyncAPIDataSet(APIDataSet):
    def _configure_session(self, session_config, _):
        return httpx.AsyncClient(**session_config)

    def _execute_request(self) -> Dict[str, Any]:
        url_dict = self._url_dict

        request_args = self._request_args
        session = self._session
        method = self._method

        name_url_list = list(url_dict.items())
        url_list = [e[1] for e in name_url_list]

        coroutines = [coroutine(session, method, url, request_args) for url in url_list]

        loop = asyncio.get_event_loop()
        tasks_done, tasks_pending = loop.run_until_complete(wait_coroutines(coroutines))

        name_list = [e[0] for e in name_url_list]
        response_dict = {}
        for name, task in zip(name_list, tasks_done):
            try:
                response_dict[name] = task.result()
            except Exception as exc:
                response_dict[name] = self._handle_exceptions(exc)

        return response_dict
