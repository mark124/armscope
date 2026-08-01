"""Bring up (and take down) the Graviton box that serves the live demo.

Graviton 3 is Neoverse V1: it has i8mm, so the demo runs the same SMMLA kernel
the whole entry is about. Ampere Altra parts, which is what the cheap European
hosts sell, are N1 and cannot.

Two sizes on purpose. The index build is embedding-bound and would take 27
hours on two cores, so it runs on a wide instance for a few hours; serving is
memory-bound and idle-heavy, so it drops to the small one afterwards. EBS
survives a type change, so this is a stop, a modify and a start, not a rebuild.

  python provision.py up          launch at build size
  python provision.py shrink      stop, drop to serve size, start again
  python provision.py status      state, type, public IP, running cost
  python provision.py down        terminate everything this script made
"""

from __future__ import annotations

import pathlib
import sys
import time

import boto3
import botocore

REGION = "us-east-2"
NAME = "armscope"
BUILD_TYPE = "m7g.2xlarge"    # 8 vCPU, 32GB, ~$0.33/h
SERVE_TYPE = "m7g.large"      # 2 vCPU,  8GB, ~$0.082/h
DISK_GB = 60
PUBKEY = pathlib.Path.home() / ".ssh" / "armscope_hetzner.pub"

ec2 = boto3.client("ec2", region_name=REGION)
res = boto3.resource("ec2", region_name=REGION)


def ubuntu_arm64() -> str:
    """Latest Ubuntu 24.04 arm64 from Canonical, by creation date."""
    imgs = ec2.describe_images(
        Owners=["099720109477"],
        Filters=[
            {"Name": "name", "Values":
                ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-arm64-server-*"]},
            {"Name": "state", "Values": ["available"]},
        ])["Images"]
    if not imgs:
        raise SystemExit("no Ubuntu 24.04 arm64 AMI found")
    newest = sorted(imgs, key=lambda i: i["CreationDate"])[-1]
    print(f"  ami {newest['ImageId']}  {newest['Name']}")
    return newest["ImageId"]


def keypair() -> str:
    """Import the key we already hold rather than having AWS mint one, so the
    private half never has to be downloaded or stored anywhere new."""
    try:
        ec2.describe_key_pairs(KeyNames=[NAME])
        print(f"  keypair {NAME} exists")
        return NAME
    except botocore.exceptions.ClientError:
        pass
    if not PUBKEY.exists():
        raise SystemExit(f"no public key at {PUBKEY}")
    ec2.import_key_pair(KeyName=NAME,
                        PublicKeyMaterial=PUBKEY.read_bytes())
    print(f"  keypair {NAME} imported from {PUBKEY.name}")
    return NAME


def security_group() -> str:
    existing = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [NAME]}])["SecurityGroups"]
    if existing:
        gid = existing[0]["GroupId"]
        print(f"  security group {gid} exists")
        return gid
    vpc = ec2.describe_vpcs(
        Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    gid = ec2.create_security_group(
        GroupName=NAME, VpcId=vpc,
        Description="armscope demo: ssh plus http/https")["GroupId"]
    ec2.authorize_security_group_ingress(
        GroupId=gid,
        IpPermissions=[
            {"IpProtocol": "tcp", "FromPort": p, "ToPort": p,
             "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}
            for p in (22, 80, 443)])
    print(f"  security group {gid} created, 22/80/443 open")
    return gid


def find():
    r = ec2.describe_instances(Filters=[
        {"Name": "tag:Name", "Values": [NAME]},
        {"Name": "instance-state-name",
         "Values": ["pending", "running", "stopping", "stopped"]}])
    for reservation in r["Reservations"]:
        for i in reservation["Instances"]:
            return i
    return None


def up() -> None:
    if (i := find()):
        print(f"already exists: {i['InstanceId']} ({i['State']['Name']})")
        return status()
    print("provisioning")
    ami, key, sg = ubuntu_arm64(), keypair(), security_group()
    inst = res.create_instances(
        ImageId=ami, InstanceType=BUILD_TYPE, MinCount=1, MaxCount=1,
        KeyName=key, SecurityGroupIds=[sg],
        BlockDeviceMappings=[{
            "DeviceName": "/dev/sda1",
            "Ebs": {"VolumeSize": DISK_GB, "VolumeType": "gp3",
                    "DeleteOnTermination": True}}],
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [{"Key": "Name", "Value": NAME}]}])[0]
    print(f"  {inst.id} starting as {BUILD_TYPE}")
    inst.wait_until_running()
    status()


def shrink() -> None:
    i = find()
    if not i:
        raise SystemExit("nothing to shrink")
    iid = i["InstanceId"]
    if i["InstanceType"] == SERVE_TYPE:
        print(f"already {SERVE_TYPE}")
        return status()
    print(f"stopping {iid}")
    ec2.stop_instances(InstanceIds=[iid])
    ec2.get_waiter("instance_stopped").wait(InstanceIds=[iid])
    ec2.modify_instance_attribute(InstanceId=iid,
                                  InstanceType={"Value": SERVE_TYPE})
    print(f"  type -> {SERVE_TYPE}, starting")
    ec2.start_instances(InstanceIds=[iid])
    ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
    print("  NOTE: the public IP changes on stop/start unless an Elastic IP "
          "is attached. Re-point DNS.")
    status()


RATE = {BUILD_TYPE: 0.3264, SERVE_TYPE: 0.0816}


def status() -> None:
    i = find()
    if not i:
        print("no instance")
        return
    state = i["State"]["Name"]
    ip = i.get("PublicIpAddress", "-")
    t = i["InstanceType"]
    print(f"{i['InstanceId']}  {t}  {state}  ip {ip}")
    if state == "running":
        hours = (time.time() - i["LaunchTime"].timestamp()) / 3600
        print(f"  up {hours:.1f}h, about ${hours * RATE.get(t, 0):.2f} so far "
              f"at ${RATE.get(t, 0):.4f}/h")
        print(f"  ssh -i ~/.ssh/armscope_hetzner ubuntu@{ip}")


def down() -> None:
    i = find()
    if i:
        ec2.terminate_instances(InstanceIds=[i["InstanceId"]])
        print(f"terminating {i['InstanceId']}")
        ec2.get_waiter("instance_terminated").wait(
            InstanceIds=[i["InstanceId"]])
    for g in ec2.describe_security_groups(
            Filters=[{"Name": "group-name",
                      "Values": [NAME]}])["SecurityGroups"]:
        ec2.delete_security_group(GroupId=g["GroupId"])
        print(f"deleted {g['GroupId']}")
    try:
        ec2.delete_key_pair(KeyName=NAME)
        print(f"deleted keypair {NAME}")
    except botocore.exceptions.ClientError:
        pass


if __name__ == "__main__":
    {"up": up, "shrink": shrink, "status": status,
     "down": down}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
