import datetime

def getCurDatetime():
    # 获取当前时间
    current_time = datetime.datetime.now()

    # 获取今天的日期
    current_date = current_time.date()

    # 获取今天是星期几（0代表星期一，1代表星期二，以此类推）
    current_weekday = current_time.weekday()

    # 将星期几转换为具体的名称
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    current_weekday_name = weekdays[current_weekday]

    # 打印结果
    # print("当前时间:", current_time)
    # print("今天的日期:", current_date)
    # print("今天是:", current_weekday_name)
    datetimeDic = {"curtime": current_time, "curdate": current_date, "curweekday": current_weekday_name}
    return datetimeDic
