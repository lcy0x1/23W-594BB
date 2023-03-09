def mem_gen(path: str, data):
    channel, time = data.size()
    ans = ""
    for i in range(time):
        for j in range(channel):
            ans += str(data[j][i].item())
        ans += "\n"
    f = open(path, "w")
    f.write(ans)
    f.close()
