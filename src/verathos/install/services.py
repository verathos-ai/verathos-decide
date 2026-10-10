"""System services: systemd units, or supervisord programs where systemd is not running (most containers).

Both restart a process that exits, start the services at boot (supervisord: whenever it starts), and keep the logs.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Service:
    name: str
    description: str
    command: list[str]
    directory: Path
    user: str
    environment: dict[str, str] = field(default_factory=dict)
    after: tuple[str, ...] = ()


def privileged(cmd: list[str]) -> list[str]:
    return cmd if os.geteuid() == 0 else ["sudo", *cmd]


def write_file(path: Path, text: str) -> None:
    """Write a file in a system directory, through sudo when not root."""
    if os.geteuid() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write(text)
    subprocess.run(["sudo", "install", "-D", "-m", "644", tmp.name, str(path)], check=True)
    os.unlink(tmp.name)


def read_file(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def systemd_quote(arg: str, *, variables: bool = True) -> str:
    """One word of an ExecStart= line, or (``variables`` False) of an Environment= line, where ``$`` is literal."""
    escaped = arg.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if variables:
        escaped = escaped.replace("$", "$$")
    return f'"{escaped}"' if not arg or any(c.isspace() or c in "\"'\\;" for c in arg) else escaped


def supervisor_quote(arg: str) -> str:
    """One word of a supervisord command= line (split like a shell) or an environment= value."""
    escaped = arg.replace("%", "%%").replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"' if not arg or any(c.isspace() or c in "\"',\\" for c in arg) else escaped


class Logs:
    """Following a service's log: ``mark`` where to start, ``lines`` what came since, ``unread`` to give lines
    back."""

    def __init__(self) -> None:
        self._pending: dict[str, list[str]] = {}

    def lines(self, name: str) -> list[str]:
        return self._pending.pop(name, []) + self.new_lines(name)

    def unread(self, name: str, lines: list[str]) -> None:
        self._pending[name] = lines + self._pending.get(name, [])

    def new_lines(self, name: str) -> list[str]:
        raise NotImplementedError

    def start_line(self) -> str | None:
        """What starts the services again after the machine restarts, where they do not start by themselves (systemd
        starts enabled units at boot)."""
        return None


class Systemd(Logs):
    name = "systemd"
    unit_dir = Path("/etc/systemd/system")

    def __init__(self) -> None:
        super().__init__()
        self._marks: dict[str, tuple[float, int]] = {}

    @staticmethod
    def render(s: Service) -> str:
        after = " ".join(["network-online.target", *(f"{a}.service" for a in s.after)])
        lines = ["[Unit]", f"Description={s.description}", f"After={after}", "Wants=network-online.target", "",
                 "[Service]", f"User={s.user}", f"WorkingDirectory={s.directory}"]
        lines += [f"Environment={systemd_quote(f'{k}={v}', variables=False)}" for k, v in sorted(s.environment.items())]
        lines += [f"ExecStart={' '.join(systemd_quote(a) for a in s.command)}", "Restart=always", "RestartSec=10",
                  "TimeoutStopSec=60", "LimitNOFILE=1048576", "", "[Install]", "WantedBy=multi-user.target", ""]
        return "\n".join(lines)

    def path(self, name: str) -> Path:
        return self.unit_dir / f"{name}.service"

    def apply(self, services: list[Service], restart: set[str]) -> set[str]:
        """Install the services, restart those new or changed and those in ``restart``, start the others if they
        are not running. Returns the names new or changed."""
        changed = set()
        for s in services:
            text = self.render(s)
            if read_file(self.path(s.name)) != text:
                write_file(self.path(s.name), text)
                changed.add(s.name)
        names = [s.name for s in services]
        subprocess.run(privileged(["systemctl", "daemon-reload"]), check=True)
        subprocess.run(privileged(["systemctl", "enable", "--quiet", *names]), check=True)
        again = [n for n in names if n in changed | restart]
        if again:
            subprocess.run(privileged(["systemctl", "restart", *again]), check=True)
        rest = [n for n in names if n not in again]
        if rest:
            subprocess.run(privileged(["systemctl", "start", *rest]), check=True)
        return changed

    def active(self, name: str) -> bool:
        return subprocess.run(["systemctl", "is-active", "--quiet", name]).returncode == 0

    def pid(self, name: str) -> int:
        """The service's process id, 0 while it is not running."""
        out = subprocess.run(["systemctl", "show", "-p", "MainPID", "--value", name], capture_output=True,
                             text=True).stdout.strip()
        return int(out) if out.isdigit() else 0

    def stop(self, names: list[str]) -> None:
        if names:
            subprocess.run(privileged(["systemctl", "stop", *names]), check=True)

    def remove(self, names: list[str]) -> None:
        present = [n for n in names if self.path(n).exists()]
        if present:
            subprocess.run(privileged(["systemctl", "disable", "--now", "--quiet", *present]))
            subprocess.run(privileged(["rm", "-f", *(str(self.path(n)) for n in present)]), check=True)
            subprocess.run(privileged(["systemctl", "daemon-reload"]), check=True)

    def mark(self, name: str) -> None:
        """Start following the service's log here: ``lines`` returns what it logs from now on."""
        self._marks[name] = (time.time(), 0)

    def new_lines(self, name: str) -> list[str]:
        since, seen = self._marks.setdefault(name, (time.time(), 0))
        out = subprocess.run(privileged(["journalctl", "-u", name, "-o", "cat", "--no-pager", "--since",
                                         f"@{int(since)}"]), capture_output=True, encoding="utf-8",
                             errors="replace").stdout.splitlines()
        self._marks[name] = (since, len(out))
        return out[seen:]

    def status_command(self, names: list[str]) -> str:
        return f"systemctl status {' '.join(names)}"

    def log_command(self, name: str) -> str:
        return f"journalctl -u {name} -f"

    def restart_command(self, names: list[str]) -> str:
        return f"systemctl restart {' '.join(names)}"


class Supervisor(Logs):
    name = "supervisor"
    conf_dir = Path("/etc/supervisor/conf.d")
    main_conf = Path("/etc/supervisor/supervisord.conf")
    log_dir = Path("/var/log/verathos")

    def __init__(self) -> None:
        super().__init__()
        self._offsets: dict[str, int] = {}

    @classmethod
    def log_path(cls, name: str) -> Path:
        return cls.log_dir / f"{name}.log"

    @classmethod
    def render(cls, s: Service) -> str:
        lines = [f"; {s.description}", f"[program:{s.name}]",
                 f"command={' '.join(supervisor_quote(a) for a in s.command)}", f"directory={s.directory}",
                 f"user={s.user}"]
        if s.environment:
            lines.append("environment=" + ",".join(f"{k}={supervisor_quote(v)}"
                                                   for k, v in sorted(s.environment.items())))
        lines += ["autostart=true", "autorestart=true", "startsecs=10", "startretries=10", "stopasgroup=true",
                  "killasgroup=true", "stopwaitsecs=60", "redirect_stderr=true",
                  f"stdout_logfile={cls.log_path(s.name)}", "stdout_logfile_maxbytes=50MB",
                  "stdout_logfile_backups=5", ""]
        return "\n".join(lines)

    def path(self, name: str) -> Path:
        return self.conf_dir / f"{name}.conf"

    def ctl(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(privileged(["supervisorctl", "-c", str(self.main_conf), *args]), capture_output=True,
                              text=True)

    def running(self) -> bool:
        return self.ctl("pid").stdout.strip().isdigit()

    def ensure_running(self) -> None:
        """Start supervisord if it is not running (a container has no init system to start it)."""
        if self.running():
            return
        subprocess.run(privileged(["supervisord", "-c", str(self.main_conf)]), check=True)
        for _ in range(50):
            if self.running():
                return
            time.sleep(0.2)
        raise SystemExit("supervisord did not start; see /var/log/supervisor/supervisord.log")

    def apply(self, services: list[Service], restart: set[str]) -> set[str]:
        """Install the services, restart those new or changed and those in ``restart``, start the others if they
        are not running. Returns the names new or changed."""
        self.ensure_running()
        subprocess.run(privileged(["mkdir", "-p", str(self.log_dir)]), check=True)
        changed = set()
        for s in services:
            text = self.render(s)
            if read_file(self.path(s.name)) != text:
                write_file(self.path(s.name), text)
                changed.add(s.name)
        names = [s.name for s in services]
        self.ctl("reread")
        self.ctl("update", *names)  # (re)starts the programs new or changed
        again = [n for n in names if n in restart and n not in changed]
        if again:
            self.ctl("restart", *again)
        for n in names:
            if not self.active(n):
                self.ctl("start", n)
        return changed

    def active(self, name: str) -> bool:
        """Running, or starting (not yet ``startsecs`` up)."""
        status = f" {self.ctl('status', name).stdout} "
        return " RUNNING " in status or " STARTING " in status

    def pid(self, name: str) -> int:
        """The program's process id, 0 while it is not running."""
        out = self.ctl("pid", name).stdout.strip()
        return int(out) if out.isdigit() else 0

    def stop(self, names: list[str]) -> None:
        if names:
            self.ctl("stop", *names)

    def remove(self, names: list[str]) -> None:
        present = [n for n in names if self.path(n).exists()]
        if not present:
            return
        if self.running():
            self.ctl("stop", *present)
        subprocess.run(privileged(["rm", "-f", *(str(self.path(n)) for n in present)]), check=True)
        if self.running():
            self.ctl("reread")
            self.ctl("update")

    def mark(self, name: str) -> None:
        """Start following the service's log here: ``lines`` returns what it logs from now on."""
        try:
            self._offsets[name] = self.log_path(name).stat().st_size
        except OSError:
            self._offsets[name] = 0

    def new_lines(self, name: str) -> list[str]:
        try:
            with open(self.log_path(name), "rb") as f:
                if f.seek(0, os.SEEK_END) < self._offsets.get(name, 0):
                    self._offsets[name] = 0  # the log was rotated
                f.seek(self._offsets.get(name, 0))
                data = f.read()
        except OSError:
            return []
        complete = data[:data.rfind(b"\n") + 1]
        self._offsets[name] = self._offsets.get(name, 0) + len(complete)
        return complete.decode(errors="replace").splitlines()

    def status_command(self, names: list[str]) -> str:
        return f"supervisorctl status {' '.join(names)}"

    def log_command(self, name: str) -> str:
        return f"tail -f {self.log_path(name)}"

    def restart_command(self, names: list[str]) -> str:
        return f"supervisorctl restart {' '.join(names)}"

    def start_line(self) -> str | None:
        """What starts the services again after the machine restarts, where they do not start by themselves."""
        return (f"after a restart of the container: supervisord -c {self.main_conf} (a container starts no "
                "services by itself)")


Manager = Systemd | Supervisor


def _environment(s: Service) -> str:
    return "".join(f"{k}={shlex.quote(v)} " for k, v in sorted(s.environment.items()) if k != "HOME")


def shell_command(s: Service) -> str:
    """The service as one shell command, for any process manager."""
    return f"cd {shlex.quote(str(s.directory))} && {_environment(s)}{shlex.join(s.command)}"


def pm2_command(s: Service) -> str:
    """The service under pm2: its program run directly (``--interpreter none``; pm2 would run a script without an
    extension it knows under node), in its directory, with its environment."""
    args = f" -- {shlex.join(s.command[1:])}" if s.command[1:] else ""
    return (f"{_environment(s)}pm2 start {shlex.quote(s.command[0])} --name {s.name} --interpreter none "
            f"--cwd {shlex.quote(str(s.directory))}{args}")


def detect() -> Manager | None:
    """systemd if it runs this machine, else supervisord if installed, else None."""
    if Path("/run/systemd/system").is_dir() and shutil.which("systemctl"):
        return Systemd()
    if shutil.which("supervisorctl") and shutil.which("supervisord"):
        return Supervisor()
    return None


def by_name(name: str | None) -> Manager | None:
    return {"systemd": Systemd, "supervisor": Supervisor}[name]() if name in ("systemd", "supervisor") else None
