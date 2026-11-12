import json

import anthropic
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.deps import LLM, Cfg, require_api_key
from app.schemas import ChatRequest

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(require_api_key)])


@router.post("/stream")
async def chat_stream(req: ChatRequest, llm: LLM, cfg: Cfg):
    async def events():
        try:
            async with llm.messages.stream(
                model=cfg.llm_model,
                max_tokens=cfg.llm_max_tokens,
                output_config={"effort": "low"},
                system="You are a concise assistant for data scientists.",
                messages=[{"role": "user", "content": req.message}],
            ) as stream:
                async for text in stream.text_stream:
                    yield f"data: {json.dumps(text)}\n\n"
                final = await stream.get_final_message()
            if final.stop_reason == "refusal":
                yield 'event: error\ndata: "request declined"\n\n'
        except anthropic.APIError as e:
            yield f"event: error\ndata: {json.dumps(e.message)}\n\n"
        except TypeError:  # the SDK raises TypeError when no credentials are configured
            yield 'event: error\ndata: "LLM credentials not configured"\n\n'
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
