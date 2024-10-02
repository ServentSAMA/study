from nonebot import on_command
from nonebot.rule import to_me
from nonebot.matcher import Matcher
from nonebot.adapters import Message
from nonebot.params import Arg, CommandArg, ArgPlainText

import datetime

wife = on_command("wife", aliases={"老婆"}, priority=10)


@wife.handle()
async def handle_wife():
    await wife.finish("谁在叫我老婆")


today = on_command("today", aliases={"日报"}, priority=10)


@today.handle()
async def handle_today():
    await wife.finish("返回日报")


kknd = on_command("kknd", aliases={"看看你的"}, priority=10)


@kknd.handle()
async def handle_kknd():
    await wife.finish("不给！！")


at = on_command("", rule=to_me())


@at.handle()
async def handle_at():
    await wife.finish("爱你😘！！")


# bg = on_command("bg",  aliases={"不给","不给！","不给！！"}, priority=10)
# @bg.handle()
# async def handle_bg():

#     await wife.finish("不给就艾草！！")

now = on_command("now", aliases={"当前", "时间"}, priority=10)


@now.handle()
async def handle_now():
    now = datetime.datetime.now()
    formatted_date = now.strftime("%Y年%m月%d日 %H时%M分%S秒")
    await wife.finish(f"当前时间为：{formatted_date}")

# @weather.handle()
# async def handle_first_receive(matcher: Matcher, args: Message = CommandArg()):
#     plain_text = args.extract_plain_text()  # 首次发送命令时跟随的参数，例：/天气 上海，则args为上海
#     print(plain_text)
#     if plain_text:
#         matcher.set_arg("city", args)  # 如果用户发送了参数则直接赋值


# @weather.got("city", prompt="你想查询哪个城市的天气呢？")
# async def handle_city(city: Message = Arg(), city_name: str = ArgPlainText("city")):
#     if city_name not in ["北京", "上海"]:  # 如果参数不符合要求，则提示用户重新输入
#         # 可以使用平台的 Message 类直接构造模板消息
#         await weather.reject(city.template("你想查询的城市 {city} 暂不支持，请重新输入！"))

#     city_weather = await get_weather(city_name)
#     await weather.finish(city_weather)


# # 在这里编写获取天气信息的函数
# async def get_weather(city: str) -> str:
#     return f"{city}的天气是..."
