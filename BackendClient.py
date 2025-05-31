import aiohttp

async def post_vlm_summary(result):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post("http://your-server-endpoint.com/api", json=result) as response:
                if response.status != 200:
                    print("POST failed with status:", response.status)
                else:
                    print("Posted successfully:", response.status)
        except aiohttp.ClientError as e:
            print("POST exception:", e)
