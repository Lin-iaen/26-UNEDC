wifi相关{
    删除已有wifi配置 : sudo nmcli connection delete "网络名称"
    连接陌生wifi : nmcli device wifi connect "网络名称" password "网络密码"
    搜索附近wifi : nmcli device wifi list
}

代理{
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
}
